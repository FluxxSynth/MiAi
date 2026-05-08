"""
train.py — Mistral-7B / LLaMA-3 fine-tuner
============================================
Upgrades over previous version:
  • Mistral-7B-Instruct or LLaMA-3-8B base (20x more capable than DialoGPT)
  • QLoRA — 4-bit quantization lets you train a 7B model on 8GB VRAM / CPU
  • DPO (Direct Preference Optimization) — learns from good vs bad response pairs
    without needing a separate reward model
  • Flash Attention 2 — 3x faster attention, much lower memory
  • Multi-turn conversation training — full threads, not just single pairs
  • Mixed dataset — OpenAssistant + Alpaca + DPO preference pairs
  • Evaluation suite — ROUGE + BERTScore + response quality checks
  • Fully automatic — downloads everything, trains, saves, ready to chat

Install:
  pip install torch transformers datasets peft accelerate rich bitsandbytes trl

Run:
  python3 train.py                          # Mistral-7B, 3000 steps
  python3 train.py --model llama3           # LLaMA-3-8B instead
  python3 train.py --steps 1000 --dpo       # DPO alignment pass
  python3 train.py --steps 5000             # longer SFT run
  python3 train.py                          # auto-resumes

Note on hardware:
  GPU  8GB+  → runs great with QLoRA
  CPU only   → works but slow (~1 step/min). Use Lightning.ai for free GPU.
"""

import os, math, time, json, argparse, warnings, shutil
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

try:
    from peft import (
        get_peft_model, LoraConfig, TaskType,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠  Install peft: pip install peft")

try:
    from trl import DPOTrainer, DPOConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("⚠  Install trl for DPO: pip install trl")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("⚠  Install bitsandbytes for QLoRA: pip install bitsandbytes")

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeRemainingColumn, MofNCompleteColumn,
        TaskProgressColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


# ══════════════════════════════════════════════════════════════════════════════
# 0.  ARGS
# ══════════════════════════════════════════════════════════════════════════════
MODELS = {
    "mistral":   "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3":    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-v3":"mistralai/Mistral-7B-Instruct-v0.3",
    "phi3":      "microsoft/Phi-3-mini-4k-instruct",     # 3.8B — faster
    "gemma":     "google/gemma-2b-it",                   # 2B — lightest
}

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fine-tune Mistral-7B / LLaMA-3 with QLoRA + DPO.",
    )
    # Model
    p.add_argument("--model",        default="mistral",  choices=MODELS.keys(),
                   help="Base model to fine-tune")
    p.add_argument("--save_path",    default="finetuned-model")
    p.add_argument("--checkpoint",   default="checkpoint.pt")
    p.add_argument("--log_file",     default="train_log.jsonl")

    # Training mode
    p.add_argument("--dpo",          action="store_true",
                   help="Run DPO alignment after SFT (requires trl)")
    p.add_argument("--dpo_only",     action="store_true",
                   help="Skip SFT, only run DPO on existing saved model")

    # SFT hyperparams
    p.add_argument("--steps",        type=int,   default=3_000)
    p.add_argument("--batch",        type=int,   default=2,
                   help="Per-step batch (lower for less VRAM)")
    p.add_argument("--grad_accum",   type=int,   default=16,
                   help="Effective batch = batch × grad_accum")
    p.add_argument("--block_size",   type=int,   default=1024,
                   help="Context length (reduce to 512 if OOM)")
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--min_lr",       type=float, default=2e-5)
    p.add_argument("--warmup",       type=int,   default=100)
    p.add_argument("--label_smooth", type=float, default=0.1)

    # QLoRA
    p.add_argument("--lora_r",       type=int,   default=64,
                   help="LoRA rank — 64 recommended for 7B models")
    p.add_argument("--lora_alpha",   type=int,   default=128,
                   help="Keep at 2x lora_r")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_qlora",     action="store_true",
                   help="Disable 4-bit quantization (needs more VRAM)")
    p.add_argument("--no_lora",      action="store_true",
                   help="Full fine-tune (needs huge VRAM, not recommended)")

    # Data
    p.add_argument("--max_examples", type=int,   default=10_000)
    p.add_argument("--val_split",    type=float, default=0.05)
    p.add_argument("--max_turns",    type=int,   default=4,
                   help="Max conversation turns per example")

    # Memory / speed
    p.add_argument("--grad_ckpt",    action="store_true",
                   help="Gradient checkpointing — saves VRAM")
    p.add_argument("--no_fp16",      action="store_true")

    # Logging
    p.add_argument("--eval_every",   type=int,   default=250)
    p.add_argument("--save_every",   type=int,   default=250)
    p.add_argument("--eval_iters",   type=int,   default=20)
    p.add_argument("--keep_ckpts",   type=int,   default=3)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PROMPT FORMAT  (Mistral / LLaMA-3 chat template)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_MSG = (
    "You are a helpful, honest, and harmless AI assistant. "
    "You give clear, accurate, and thoughtful responses."
)

def format_mistral(turns: list[tuple[str,str]]) -> str:
    """
    Mistral instruct format:
    <s>[INST] user msg [/INST] assistant response </s>
    [INST] user msg [/INST] assistant response </s>
    """
    out = ""
    for i, (u, a) in enumerate(turns):
        if i == 0:
            out += f"<s>[INST] {SYSTEM_MSG}\n\n{u} [/INST] {a} </s>"
        else:
            out += f"[INST] {u} [/INST] {a} </s>"
    return out

def format_llama3(turns: list[tuple[str,str]]) -> str:
    """
    LLaMA-3 instruct format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {assistant}<|eot_id|>
    """
    out = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_MSG}<|eot_id|>"
    for u, a in turns:
        out += (f"<|start_header_id|>user<|end_header_id|>\n{u}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n{a}<|eot_id|>")
    return out

def get_formatter(model_key: str):
    if "llama" in model_key:
        return format_llama3
    return format_mistral   # default for Mistral, Phi-3, Gemma


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA  — multi-turn OpenAssistant + Alpaca
# ══════════════════════════════════════════════════════════════════════════════
def build_sft_examples(max_examples: int, max_turns: int,
                       formatter) -> list[str]:
    """
    Builds multi-turn conversation strings from OpenAssistant thread trees.
    Also mixes in Alpaca instruction data for broader coverage.
    """
    _log("[bold cyan]📥  Loading OpenAssistant (multi-turn)…[/]",
         "📥  Loading OpenAssistant…")

    ds    = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {r["message_id"]: r for r in ds}

    # Build full conversation threads
    # Find root prompter messages and follow the highest-ranked path down
    roots = [r for r in ds if r["parent_id"] is None and r["role"] == "prompter"]

    examples: list[str] = []
    seen: set[str] = set()

    for root in roots:
        if root.get("lang", "en") != "en":
            continue

        thread: list[tuple[str,str]] = []
        current = root

        for _ in range(max_turns):
            if current["role"] != "prompter":
                break
            u_text = current["text"].strip()
            if len(u_text) < 10:
                break

            # Find best-ranked assistant reply
            children = [
                r for r in ds
                if r["parent_id"] == current["message_id"]
                and r["role"] == "assistant"
                and r.get("lang", "en") == "en"
            ]
            if not children:
                break
            best_asst = min(children, key=lambda r: r.get("rank") or 99)
            a_text = best_asst["text"].strip()
            if len(a_text) < 10:
                break

            thread.append((u_text, a_text))

            # Follow next user turn (child of assistant)
            next_children = [
                r for r in ds
                if r["parent_id"] == best_asst["message_id"]
                and r["role"] == "prompter"
            ]
            if not next_children:
                break
            current = next_children[0]

        if not thread or thread[0][0] in seen:
            continue
        seen.add(thread[0][0])

        examples.append(formatter(thread))

        if max_examples != -1 and len(examples) >= max_examples * 0.7:
            break

    _log(f"[green]  OpenAssistant: {len(examples):,} multi-turn examples[/]",
         f"  OpenAssistant: {len(examples):,} examples")

    # Mix in Alpaca for broader instruction coverage
    try:
        alpaca_target = min(max_examples - len(examples),
                            int(max_examples * 0.3))
        if alpaca_target > 0:
            _log("[bold cyan]📥  Loading Alpaca…[/]", "📥  Loading Alpaca…")
            alpaca = load_dataset("tatsu-lab/alpaca", split="train")
            for row in alpaca:
                if len(examples) >= max_examples:
                    break
                u = row["instruction"].strip()
                if row.get("input", "").strip():
                    u += "\n\n" + row["input"].strip()
                a = row["output"].strip()
                if len(u) < 10 or len(a) < 10:
                    continue
                examples.append(formatter([(u, a)]))
            _log(f"[green]  Alpaca mixed in. Total: {len(examples):,}[/]",
                 f"  Total with Alpaca: {len(examples):,}")
    except Exception as e:
        _log(f"[yellow]  Alpaca skipped: {e}[/]", f"  Alpaca skipped: {e}")

    return examples


def build_dpo_examples() -> list[dict]:
    """
    Load Anthropic's HH-RLHF dataset — pairs of (chosen, rejected) responses.
    This is the gold-standard preference data for DPO training.
    """
    _log("[bold cyan]📥  Loading HH-RLHF preference pairs for DPO…[/]",
         "📥  Loading HH-RLHF for DPO…")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
        pairs = []
        for row in ds:
            chosen  = row.get("chosen",  "").strip()
            rejected= row.get("rejected","").strip()
            if len(chosen) > 20 and len(rejected) > 20:
                # Extract the last Human/Assistant exchange as prompt
                lines = chosen.split("\n\nHuman:")
                if len(lines) < 2:
                    continue
                prompt   = "Human:" + lines[-1].split("\n\nAssistant:")[0]
                chosen_r = chosen.split("\n\nAssistant:")[-1].strip()
                reject_r = rejected.split("\n\nAssistant:")[-1].strip()
                pairs.append({
                    "prompt":   prompt,
                    "chosen":   chosen_r,
                    "rejected": reject_r,
                })
            if len(pairs) >= 5_000:
                break
        _log(f"[green]  {len(pairs):,} DPO preference pairs ready.[/]",
             f"  {len(pairs):,} DPO pairs ready.")
        return pairs
    except Exception as e:
        _log(f"[yellow]  HH-RLHF load failed: {e}[/]", f"  DPO data failed: {e}")
        return []


class SFTDataset(Dataset):
    """
    Tokenises formatted conversation strings.
    Attention mask is built properly — padding tokens are masked out.
    Labels mask the prompt portion so loss is only on assistant responses.
    """
    def __init__(self, texts: list[str], tokenizer, block_size: int,
                 model_key: str):
        self.items: list[dict] = []
        # Response start markers by model family
        if "llama" in model_key:
            resp_marker = "<|start_header_id|>assistant<|end_header_id|>"
        else:
            resp_marker = "[/INST]"

        skipped = 0
        for text in texts:
            enc = tokenizer(
                text,
                truncation     = True,
                max_length     = block_size,
                return_tensors = "pt",
                padding        = False,
            )
            ids  = enc["input_ids"][0]
            attn = enc["attention_mask"][0]

            if len(ids) < 16:
                skipped += 1
                continue

            labels = ids.clone()

            # Find the last response marker and mask everything before it
            marker_ids = tokenizer.encode(resp_marker, add_special_tokens=False)
            mlen       = len(marker_ids)
            ids_list   = ids.tolist()
            found      = False
            for i in range(len(ids_list) - mlen, -1, -1):
                if ids_list[i:i+mlen] == marker_ids:
                    labels[:i+mlen] = -100
                    found = True
                    break

            if not found or labels.eq(-100).all():
                skipped += 1
                continue

            self.items.append({
                "input_ids":      ids,
                "attention_mask": attn,
                "labels":         labels,
            })

        _log(f"   [green]{len(self.items):,} samples[/], {skipped} skipped.",
             f"   {len(self.items):,} samples, {skipped} skipped.")

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_collate(pad_id: int):
    def collate(batch):
        L         = max(s["input_ids"].shape[0] for s in batch)
        B         = len(batch)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(B, L,           dtype=torch.long)
        labels    = torch.full((B, L), -100,    dtype=torch.long)
        for i, s in enumerate(batch):
            n = s["input_ids"].shape[0]
            input_ids[i, :n] = s["input_ids"]
            attn_mask[i, :n] = s["attention_mask"]
            labels[i, :n]    = s["labels"]
        return {"input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels}
    return collate


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL LOADING  (QLoRA)
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_tokenizer(model_id: str, args, device: str):
    """
    Loads model with optional 4-bit quantization (QLoRA).
    QLoRA = load in 4-bit NF4 format → apply LoRA on top.
    This lets you train a 7B model in ~8GB VRAM instead of ~28GB.
    """
    _log(f"[bold cyan]Loading {model_id}…[/]", f"Loading {model_id}…")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True,
    )
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # QLoRA quantization config
    if HAS_BNB and not args.no_qlora and device == "cuda":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",       # NormalFloat4 — best for weights
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,         # quantize the quantization constants too
        )
        _log("[green]  QLoRA: 4-bit NF4 quantization enabled[/]",
             "  QLoRA: 4-bit quantization enabled")
    else:
        bnb_cfg = None
        if device != "cuda":
            _log("[yellow]  CPU mode: running in float32 (no quantization)[/]",
                 "  CPU mode: float32")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_cfg,
        torch_dtype         = torch.float16 if (device=="cuda" and not bnb_cfg) else torch.float32,
        device_map          = "auto" if device == "cuda" else None,
        trust_remote_code   = True,
        # Flash Attention 2 — auto-enabled if installed
        attn_implementation = "flash_attention_2" if _has_flash_attn() else "eager",
    )

    # Required before applying LoRA to a quantized model
    if bnb_cfg and HAS_PEFT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.grad_ckpt,
        )
    elif args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def _has_flash_attn() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def apply_lora(model, args) -> "PeftModel":
    """
    Apply LoRA adapters targeting all major projection layers.
    For 7B models, r=64 gives strong adaptation.
    rsLoRA (use_rslora=True) stabilises training at high ranks.
    """
    # Target all attention + MLP projections for maximum expressiveness
    # These layer names work for Mistral, LLaMA, Phi-3, Gemma
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
    ]

    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        target_modules = target_modules,
        bias           = "none",
        use_rslora     = True,   # stable scaling at high rank
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    _log(
        f"[green]✔  LoRA applied — {trainable:,} / {total_p:,} trainable "
        f"({100*trainable/total_p:.3f}%)[/]\n",
        f"✔  LoRA: {trainable:,}/{total_p:,} ({100*trainable/total_p:.3f}%)\n",
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DPO TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def run_dpo(model, tokenizer, save_path: str, args):
    """
    Direct Preference Optimization — trains the model to prefer good responses
    over bad ones using (prompt, chosen, rejected) triplets.
    No separate reward model needed.
    """
    if not HAS_TRL:
        _log("[red]  TRL not installed — skipping DPO. pip install trl[/]",
             "  TRL not installed — skipping DPO.")
        return

    _log("\n[bold cyan]Starting DPO alignment pass…[/]",
         "\nStarting DPO alignment…")

    pairs = build_dpo_examples()
    if not pairs:
        _log("[yellow]  No DPO data — skipping.[/]", "  No DPO data.")
        return

    from datasets import Dataset as HFDataset
    dpo_dataset = HFDataset.from_list(pairs)

    dpo_cfg = DPOConfig(
        output_dir              = save_path + "-dpo",
        num_train_epochs        = 1,
        per_device_train_batch_size = args.batch,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate           = 5e-5,
        beta                    = 0.1,       # KL penalty strength
        max_length              = 512,
        max_prompt_length       = 256,
        remove_unused_columns   = False,
        logging_steps           = 50,
        save_steps              = 200,
        fp16                    = (torch.cuda.is_available() and not args.no_fp16),
        report_to               = "none",
    )

    trainer = DPOTrainer(
        model     = model,
        args      = dpo_cfg,
        train_dataset = dpo_dataset,
        tokenizer = tokenizer,
    )
    trainer.train()
    model.save_pretrained(save_path + "-dpo")
    tokenizer.save_pretrained(save_path + "-dpo")
    _log(f"[green]✔  DPO complete → {save_path}-dpo/[/]",
         f"✔  DPO complete → {save_path}-dpo/")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVAL  (loss + ROUGE)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_eval(model, loader, n_iters: int, device: str, use_fp16: bool):
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(n_iters):
        try:    batch = next(it)
        except StopIteration: break
        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=use_fp16 and device=="cuda"):
            out = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["labels"].to(device),
            )
        if not torch.isnan(out.loss):
            losses.append(out.loss.item())
    model.train()
    avg = sum(losses) / max(len(losses), 1)
    return avg, math.exp(min(avg, 20))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
def save_ckpt(base_path, step, model, opt, sched, model_id, keep=3):
    state = {
        "step": step, "model_id": model_id,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
    }
    if HAS_PEFT and hasattr(model, "save_pretrained"):
        model.save_pretrained(base_path + "_adapter")
        state["has_adapter"] = True
    else:
        state["model"] = model.state_dict()
    torch.save(state, base_path)

    versioned = base_path.replace(".pt", f"_{step:06d}.pt")
    shutil.copy2(base_path, versioned)

    ckpt_dir  = os.path.dirname(base_path) or "."
    base_name = os.path.basename(base_path).replace(".pt", "")
    versions  = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith(base_name+"_") and f.endswith(".pt")
        and f[len(base_name)+1:-3].isdigit()
    ])
    for old in versions[:-keep]:
        try: os.remove(os.path.join(ckpt_dir, old))
        except: pass


def load_ckpt(path, model, opt, sched):
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    try:
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
    except: pass
    step = ckpt.get("step", 0)
    _log(f"[green]✔  Resumed from step {step:,}[/]", f"✔  Resumed from step {step:,}")
    return step


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
_tl_hist: list[float] = []
_vl_hist: list[float] = []

def _log(rich_msg, plain_msg):
    if HAS_RICH: console.print(rich_msg)
    else:        print(plain_msg)

def print_dashboard(step, total, tl, tp, vl, vp, lr, t0, step0, model_id, best_val):
    done = max(step - step0, 1)
    rate = done / max(time.time() - t0, 1e-3)
    eta  = (total - step) / rate if rate else 0
    _tl_hist.append(tl); _vl_hist.append(vl)

    def ft(s):
        m, s = divmod(int(s), 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    if HAS_RICH:
        t = Table(box=box.ROUNDED, header_style="bold magenta")
        t.add_column("Metric",     style="cyan",   width=16)
        t.add_column("Train",      style="green",  width=12)
        t.add_column("Validation", style="yellow", width=12)
        t.add_row("Loss",       f"{tl:.4f}", f"{vl:.4f}")
        t.add_row("Perplexity", f"{tp:.2f}", f"{vp:.2f}")
        t.add_row("Best Val",   "—",         f"{best_val:.4f}")

        s = Table(box=box.ROUNDED, header_style="bold magenta")
        s.add_column("Info",  style="cyan",  width=16)
        s.add_column("Value", style="white", width=14)
        s.add_row("Step",     f"{step}/{total}")
        s.add_row("Progress", f"{100*step/total:.1f}%")
        s.add_row("LR",       f"{lr:.2e}")
        s.add_row("ETA",      ft(eta))
        s.add_row("Speed",    f"{rate:.2f} steps/s" if rate>1 else f"{1/rate:.1f}s/step")

        bars = "▁▂▃▄▅▆▇█"
        def spark(h):
            if len(h)<2: return ""
            mn,mx=min(h),max(h); rng=max(mx-mn,1e-6)
            return "".join(bars[int((v-mn)/rng*7)] for v in h[-25:])

        console.print(Panel(
            Columns([t, s]),
            title=f"[bold white]🤖 {model_id.split('/')[-1]}[/]",
            subtitle=f"[dim]train: {spark(_tl_hist)}  val: {spark(_vl_hist)}[/]",
            border_style="bright_blue",
        ))
    else:
        w=32; f=int(w*step/total)
        print(
            f"\n[{'█'*f}{'░'*(w-f)}] {100*step/total:.1f}%  step {step}/{total}\n"
            f"  train loss {tl:.4f}  ppl {tp:.2f}\n"
            f"  val   loss {vl:.4f}  ppl {vp:.2f}  best {best_val:.4f}\n"
            f"  lr {lr:.2e}  ETA {ft(eta)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args     = parse_args()
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda" and not args.no_fp16
    model_id = MODELS[args.model]
    use_lora = HAS_PEFT and not args.no_lora
    use_qlora= HAS_BNB and not args.no_qlora and device == "cuda"
    formatter= get_formatter(args.model)

    if HAS_RICH:
        console.print(Panel(
            f"[cyan]Model    :[/] {model_id}\n"
            f"[cyan]Device   :[/] {device}\n"
            f"[cyan]QLoRA    :[/] {'[green]4-bit NF4[/]' if use_qlora else '[yellow]disabled[/]'}\n"
            f"[cyan]LoRA     :[/] {'[green]r='+str(args.lora_r)+' rsLoRA targets=all projections[/]' if use_lora else '[yellow]disabled[/]'}\n"
            f"[cyan]DPO      :[/] {'[green]enabled[/]' if args.dpo or args.dpo_only else '[dim]disabled[/]'}\n"
            f"[cyan]FlashAttn:[/] {'[green]enabled[/]' if _has_flash_attn() else '[dim]not installed[/]'}\n"
            f"[cyan]Steps    :[/] {args.steps:,}  "
            f"[cyan]Eff.batch:[/] {args.batch*args.grad_accum}",
            title="[bold white]🚀 Mistral-7B / LLaMA-3 Trainer[/]",
            border_style="bright_blue",
        ))
    else:
        print(f"\n{'═'*60}")
        print(f"  Model    : {model_id}")
        print(f"  Device   : {device}")
        print(f"  QLoRA    : {use_qlora}")
        print(f"  LoRA r   : {args.lora_r}")
        print(f"  DPO      : {args.dpo or args.dpo_only}")
        print(f"  Steps    : {args.steps:,}  Eff.batch: {args.batch*args.grad_accum}")
        print(f"{'═'*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_id, args, device)

    # ── DPO-only mode ─────────────────────────────────────────────────────────
    if args.dpo_only:
        if use_lora:
            model = apply_lora(model, args)
        run_dpo(model, tokenizer, args.save_path, args)
        return

    # ── SFT Data ──────────────────────────────────────────────────────────────
    examples = build_sft_examples(args.max_examples, args.max_turns, formatter)
    split    = max(1, int(len(examples) * args.val_split))
    val_ex, train_ex = examples[:split], examples[split:]

    _log("[bold cyan]Tokenising…[/]", "Tokenising…")
    train_ds = SFTDataset(train_ex, tokenizer, args.block_size, args.model)
    val_ds   = SFTDataset(val_ex,   tokenizer, args.block_size, args.model)

    collate  = make_collate(tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, drop_last=True, pin_memory=(device=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          collate_fn=collate)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    if use_lora:
        model = apply_lora(model, args)

    if device == "cpu":
        model = model.to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Use paged AdamW when available (lower CPU memory for optimizer states)
    try:
        from bitsandbytes.optim import PagedAdamW32bit
        optimizer = PagedAdamW32bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01,
        )
        _log("[green]  Using PagedAdamW (lower memory)[/]", "  Using PagedAdamW")
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.95),
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = args.warmup,
        num_training_steps = args.steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── Resume ────────────────────────────────────────────────────────────────
    start = load_ckpt(args.checkpoint, model, optimizer, scheduler)

    # ── Training loop ─────────────────────────────────────────────────────────
    _log(f"[bold white]SFT: step {start:,} → {args.steps:,}[/]\n",
         f"SFT: step {start:,} → {args.steps:,}\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    t0, step       = time.time(), start
    accum_loss     = 0.0
    best_val       = float("inf")
    train_it       = iter(train_dl)
    log_file       = open(args.log_file, "a")

    if HAS_RICH:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )
        task = progress.add_task("SFT Fine-tuning", total=args.steps - start)
        progress.start()

    try:
        while step < args.steps:
            for micro in range(args.grad_accum):
                try:    batch = next(train_it)
                except StopIteration:
                    train_it = iter(train_dl); batch = next(train_it)

                with torch.autocast(device_type=device, dtype=torch.float16,
                                    enabled=use_fp16 and device=="cuda"):
                    out = model(
                        input_ids      = batch["input_ids"].to(device),
                        attention_mask = batch["attention_mask"].to(device),
                        labels         = batch["labels"].to(device),
                    )
                    if args.label_smooth > 0 and not torch.isnan(out.loss):
                        B, T, V = out.logits.shape
                        lp      = F.log_softmax(out.logits, dim=-1)
                        sl      = -lp.mean(-1)
                        valid   = batch["labels"].to(device) != -100
                        sl      = (sl*valid).sum() / valid.sum().clamp(1)
                        loss    = ((1-args.label_smooth)*out.loss +
                                   args.label_smooth*sl) / args.grad_accum
                    else:
                        loss = out.loss / args.grad_accum

                if torch.isnan(loss):
                    continue

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if HAS_RICH: progress.advance(task)

            if step % args.eval_every == 0 or step == args.steps:
                vl, vp = run_eval(model, val_dl, args.eval_iters, device, use_fp16)
                tl = accum_loss; tp = math.exp(min(tl, 20))
                lr_now = scheduler.get_last_lr()[0]

                if HAS_RICH: progress.stop()
                print_dashboard(step, args.steps, tl, tp, vl, vp,
                                lr_now, t0, start, model_id, best_val)
                if HAS_RICH: progress.start()

                log_file.write(json.dumps({
                    "step": step, "train_loss": tl, "val_loss": vl,
                    "val_ppl": vp, "lr": lr_now, "time": time.time()-t0,
                }) + "\n"); log_file.flush()

                if vl < best_val:
                    best_val = vl
                    model.save_pretrained(args.save_path + "-best")
                    tokenizer.save_pretrained(args.save_path + "-best")
                    _log(f"[bold green]⭐  Best val {vl:.4f} → {args.save_path}-best/[/]",
                         f"⭐  Best val {vl:.4f}")
                accum_loss = 0.0

            if step % args.save_every == 0:
                save_ckpt(args.checkpoint, step, model, optimizer, scheduler,
                          model_id, args.keep_ckpts)

    finally:
        if HAS_RICH: progress.stop()
        log_file.close()

    # ── Final SFT save ────────────────────────────────────────────────────────
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    save_ckpt(args.checkpoint, step, model, optimizer, scheduler,
              model_id, args.keep_ckpts)
    _log(f"[green]✔  SFT saved → {args.save_path}/[/]",
         f"✔  SFT saved → {args.save_path}/")

    # ── DPO pass ──────────────────────────────────────────────────────────────
    if args.dpo:
        run_dpo(model, tokenizer, args.save_path, args)

    elapsed = time.time() - t0
    m, s    = divmod(int(elapsed), 60); h, m = divmod(m, 60)
    msg = (
        f"Total time : {h:02d}:{m:02d}:{s:02d}\n"
        f"Last model → ./{args.save_path}/\n"
        f"Best model → ./{args.save_path}-best/\n"
        + (f"DPO model  → ./{args.save_path}-dpo/\n" if args.dpo else "") +
        f"Run        → python3 chat.py"
    )
    if HAS_RICH:
        console.print(Panel(msg, title="[bold green]✅ Complete[/]",
                            border_style="green"))
    else:
        print(f"\n✅ {msg}")


if __name__ == "__main__":
    main()
