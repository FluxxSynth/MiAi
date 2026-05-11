"""
train.py — Mistral-7B / LLaMA-3 fine-tuner
============================================
Improvements over original:
  • Early stopping — halts training when val loss stops improving
  • DPO data validation — logs parse success/failure rate, filters bad pairs
  • All original features preserved (QLoRA, DPO, Flash Attention, rich dashboard)

Install:
    pip install torch transformers datasets peft accelerate rich bitsandbytes trl

Run:
    python3 train.py                    # Mistral-7B, 3000 steps
    python3 train.py --model llama3     # LLaMA-3-8B
    python3 train.py --steps 1000 --dpo # with DPO alignment
    python3 train.py --early_stop 5     # stop if no improvement for 5 evals
"""

import os, math, time, json, argparse, warnings, shutil
warnings.filterwarnings("once", category=UserWarning,
                        module="(transformers|torch|peft|datasets)")

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
        prepare_model_for_kbit_training, PeftModel,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠ Install peft: pip install peft")

try:
    from trl import DPOTrainer, DPOConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("⚠ Install trl for DPO: pip install trl")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("⚠ Install bitsandbytes for QLoRA: pip install bitsandbytes")

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn,
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
# 0. ARGS
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "mistral":    "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3":     "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-v3": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3":       "microsoft/Phi-3-mini-4k-instruct",
    "gemma":      "google/gemma-2b-it",
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fine-tune Mistral-7B / LLaMA-3 with QLoRA + DPO.",
    )
    # Model
    p.add_argument("--model",      default="mistral", choices=MODELS.keys())
    p.add_argument("--save_path",  default="finetuned-model")
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--log_file",   default="train_log.jsonl")

    # Training mode
    p.add_argument("--dpo",      action="store_true",
                   help="Run DPO alignment after SFT")
    p.add_argument("--dpo_only", action="store_true",
                   help="Skip SFT, only run DPO on existing saved model")

    # SFT hyperparams
    p.add_argument("--steps",        type=int,   default=3_000)
    p.add_argument("--batch",        type=int,   default=2)
    p.add_argument("--grad_accum",   type=int,   default=16)
    p.add_argument("--block_size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--min_lr",       type=float, default=2e-5)
    p.add_argument("--warmup",       type=int,   default=100)
    p.add_argument("--label_smooth", type=float, default=0.1)

    # QLoRA
    p.add_argument("--lora_r",       type=int,   default=64)
    p.add_argument("--lora_alpha",   type=int,   default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_qlora",     action="store_true")
    p.add_argument("--no_lora",      action="store_true")

    # Data
    p.add_argument("--max_examples", type=int,   default=10_000)
    p.add_argument("--val_split",    type=float, default=0.05)
    p.add_argument("--max_turns",    type=int,   default=4)

    # Memory / speed
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--no_fp16",   action="store_true")

    # Logging
    p.add_argument("--eval_every",  type=int, default=250)
    p.add_argument("--save_every",  type=int, default=250)
    p.add_argument("--eval_iters",  type=int, default=20)
    p.add_argument("--keep_ckpts", type=int, default=3)

    # ── Early stopping (improvement) ─────────────────────────────────────
    p.add_argument("--early_stop", type=int, default=0,
                   help="Stop if val loss doesn't improve for N eval intervals. "
                        "0 = disabled.")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROMPT FORMAT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_MSG = (
    "You are a helpful, honest, and harmless AI assistant. "
    "You give clear, accurate, and thoughtful responses."
)


def format_mistral(turns: list[tuple[str, str]]) -> str:
    out = ""
    for i, (u, a) in enumerate(turns):
        if i == 0:
            out += f"<s>[INST] {SYSTEM_MSG}\n\n{u} [/INST] {a} </s>"
        else:
            out += f"[INST] {u} [/INST] {a} </s>"
    return out


def format_llama3(turns: list[tuple[str, str]]) -> str:
    out = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
           f"\n{SYSTEM_MSG}<|eot_id|>")
    for u, a in turns:
        out += (f"<|start_header_id|>user<|end_header_id|>\n{u}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n{a}<|eot_id|>")
    return out


def get_formatter(model_key: str):
    return format_llama3 if "llama" in model_key else format_mistral


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA
# ══════════════════════════════════════════════════════════════════════════════

def build_sft_examples(max_examples: int, max_turns: int, formatter) -> list[str]:
    _log("[bold cyan]📥 Loading OpenAssistant (multi-turn)…[/]",
         "📥 Loading OpenAssistant…")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    roots    = [r for r in ds if r["parent_id"] is None and r["role"] == "prompter"]
    examples : list[str] = []
    seen     : set[str]  = set()

    for root in roots:
        if root.get("lang", "en") != "en":
            continue
        thread: list[tuple[str, str]] = []
        current = root

        for _ in range(max_turns):
            if current["role"] != "prompter":
                break
            u_text = current["text"].strip()
            if len(u_text) < 10:
                break

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

    _log(f"[green]  OpenAssistant: {len(examples):,} examples[/]",
         f"  OpenAssistant: {len(examples):,} examples")

    try:
        alpaca_target = min(max_examples - len(examples), int(max_examples * 0.3))
        if alpaca_target > 0:
            _log("[bold cyan]📥 Loading Alpaca…[/]", "📥 Loading Alpaca…")
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
            _log(f"[green]  Total: {len(examples):,}[/]",
                 f"  Total with Alpaca: {len(examples):,}")
    except Exception as e:
        _log(f"[yellow]  Alpaca skipped: {e}[/]", f"  Alpaca skipped: {e}")

    return examples


def build_dpo_examples() -> list[dict]:
    """
    Load HH-RLHF preference pairs with validation logging.

    Improvements:
      • Counts total/parsed/skipped so you can see data quality at a glance.
      • Filters pairs where chosen == rejected (degenerate).
      • Filters pairs where either side is suspiciously short (<20 chars).
      • Validates that both chosen and rejected actually contain an
        Assistant turn before accepting the pair.
    """
    _log("[bold cyan]📥 Loading HH-RLHF preference pairs for DPO…[/]",
         "📥 Loading HH-RLHF for DPO…")

    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
    except Exception as e:
        _log(f"[red]  HH-RLHF load failed: {e}[/]", f"  HH-RLHF load failed: {e}")
        return []

    pairs: list[dict] = []
    stats = {"total": 0, "no_human_split": 0, "no_asst_split": 0,
             "too_short": 0, "identical": 0, "ok": 0}

    for row in ds:
        stats["total"] += 1
        chosen   = (row.get("chosen")   or "").strip()
        rejected = (row.get("rejected") or "").strip()

        if not chosen or not rejected:
            stats["no_human_split"] += 1
            continue

        # Must have at least one Human/Assistant exchange
        if "\n\nHuman:" not in chosen:
            stats["no_human_split"] += 1
            continue

        # Extract prompt (everything up to last Human turn)
        lines = chosen.split("\n\nHuman:")
        prompt_raw = "Human:" + lines[-1]

        if "\n\nAssistant:" not in prompt_raw:
            stats["no_asst_split"] += 1
            continue

        # Validate rejected has the same structure
        if "\n\nAssistant:" not in rejected:
            stats["no_asst_split"] += 1
            continue

        chosen_r   = chosen.split("\n\nAssistant:")[-1].strip()
        rejected_r = rejected.split("\n\nAssistant:")[-1].strip()
        prompt     = prompt_raw.split("\n\nAssistant:")[0].strip()

        # Filter degenerate pairs
        if len(chosen_r) < 20 or len(rejected_r) < 20:
            stats["too_short"] += 1
            continue
        if chosen_r == rejected_r:
            stats["identical"] += 1
            continue

        pairs.append({
            "prompt":   prompt,
            "chosen":   chosen_r,
            "rejected": rejected_r,
        })
        stats["ok"] += 1

        if len(pairs) >= 5_000:
            break

    # ── Validation report ─────────────────────────────────────────────────
    total = max(stats["total"], 1)
    parse_rate = 100 * stats["ok"] / total
    _log(
        f"[green]  DPO pairs: {stats['ok']:,} / {stats['total']:,} "
        f"({parse_rate:.1f}% usable)[/]\n"
        f"  [dim]skipped — no_human_split: {stats['no_human_split']}, "
        f"no_asst_split: {stats['no_asst_split']}, "
        f"too_short: {stats['too_short']}, "
        f"identical: {stats['identical']}[/]",
        f"  DPO pairs: {stats['ok']:,}/{stats['total']:,} ({parse_rate:.1f}%) | "
        f"skipped: no_human={stats['no_human_split']} no_asst={stats['no_asst_split']} "
        f"short={stats['too_short']} dup={stats['identical']}",
    )

    if parse_rate < 30:
        _log("[yellow]  ⚠ Low parse rate — dataset format may have changed.[/]",
             "  ⚠ Low parse rate on HH-RLHF.")

    return pairs


class SFTDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, block_size: int, model_key: str):
        self.items: list[dict] = []
        resp_marker = (
            "<|start_header_id|>assistant<|end_header_id|>"
            if "llama" in model_key else "[/INST]"
        )
        skipped = 0
        for text in texts:
            enc = tokenizer(
                text, truncation=True, max_length=block_size,
                return_tensors="pt", padding=False,
            )
            ids  = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            if len(ids) < 16:
                skipped += 1
                continue

            labels       = ids.clone()
            marker_ids   = tokenizer.encode(resp_marker, add_special_tokens=False)
            mlen         = len(marker_ids)
            ids_list     = ids.tolist()
            found        = False
            for i in range(len(ids_list) - mlen, -1, -1):
                if ids_list[i:i + mlen] == marker_ids:
                    labels[:i + mlen] = -100
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

        _log(f"  [green]{len(self.items):,} samples[/], {skipped} skipped.",
             f"  {len(self.items):,} samples, {skipped} skipped.")

    def __len__(self):           return len(self.items)
    def __getitem__(self, i):    return self.items[i]


def make_collate(pad_id: int):
    def collate(batch):
        L = max(s["input_ids"].shape[0] for s in batch)
        B = len(batch)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(B, L, dtype=torch.long)
        labels    = torch.full((B, L), -100, dtype=torch.long)
        for i, s in enumerate(batch):
            n = s["input_ids"].shape[0]
            input_ids[i, :n] = s["input_ids"]
            attn_mask[i, :n] = s["attention_mask"]
            labels[i, :n]    = s["labels"]
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}
    return collate


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL LOADING (QLoRA)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_id: str, args, device: str):
    _log(f"[bold cyan]Loading {model_id}…[/]", f"Loading {model_id}…")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_cfg = None
    if HAS_BNB and not args.no_qlora and device == "cuda":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _log("[green]  QLoRA: 4-bit NF4 enabled[/]", "  QLoRA: 4-bit enabled")
    else:
        if device != "cuda":
            _log("[yellow]  CPU mode: float32[/]", "  CPU mode: float32")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16 if (device == "cuda" and not bnb_cfg) else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "eager",
    )

    if bnb_cfg and HAS_PEFT:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_ckpt
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


def apply_lora(model, args):
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    _log(
        f"[green]✔ LoRA: {trainable:,}/{total_p:,} ({100*trainable/total_p:.3f}%)[/]",
        f"✔ LoRA: {trainable:,}/{total_p:,} ({100*trainable/total_p:.3f}%)",
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. DPO
# ══════════════════════════════════════════════════════════════════════════════

def run_dpo(model, tokenizer, save_path: str, args):
    if not HAS_TRL:
        _log("[red]  TRL not installed. pip install trl[/]", "  TRL not installed.")
        return

    _log("\n[bold cyan]Starting DPO alignment pass…[/]", "\nStarting DPO…")
    pairs = build_dpo_examples()
    if not pairs:
        _log("[yellow]  No DPO data.[/]", "  No DPO data.")
        return

    from datasets import Dataset as HFDataset
    dpo_dataset = HFDataset.from_list(pairs)

    dpo_cfg = DPOConfig(
        output_dir=save_path + "-dpo",
        num_train_epochs=1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=5e-5,
        beta=0.1,
        max_length=512,
        max_prompt_length=256,
        remove_unused_columns=False,
        logging_steps=50,
        save_steps=200,
        fp16=(torch.cuda.is_available() and not args.no_fp16),
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(save_path + "-dpo")
    tokenizer.save_pretrained(save_path + "-dpo")
    _log(f"[green]✔ DPO complete → {save_path}-dpo/[/]",
         f"✔ DPO complete → {save_path}-dpo/")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVAL
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_eval(model, loader, n_iters: int, device: str, use_fp16: bool):
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(n_iters):
        try:
            batch = next(it)
        except StopIteration:
            break
        with torch.autocast(device_type=device, dtype=torch.float16,
                             enabled=use_fp16 and device == "cuda"):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
        if not torch.isnan(out.loss):
            losses.append(out.loss.item())
    model.train()
    avg = sum(losses) / max(len(losses), 1)
    return avg, math.exp(min(avg, 20))


# ══════════════════════════════════════════════════════════════════════════════
# 6. EARLY STOPPING  (improvement)
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training if validation loss doesn't improve for `patience` evals.

    Usage:
        es = EarlyStopping(patience=5, min_delta=1e-4)
        ...
        if es(val_loss):
            break  # stop training
    """

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("inf")
        self.wait       = 0
        self.triggered  = False

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if self.patience == 0:
            return False

        if val_loss < self.best - self.min_delta:
            self.best  = val_loss
            self.wait  = 0
        else:
            self.wait += 1
            _log(
                f"[yellow]  Early stop counter: {self.wait}/{self.patience}[/]",
                f"  Early stop counter: {self.wait}/{self.patience}",
            )
            if self.wait >= self.patience:
                self.triggered = True
                _log(
                    f"[bold red]  ⏹ Early stopping triggered "
                    f"(no improvement for {self.patience} evals). "
                    f"Best val loss: {self.best:.4f}[/]",
                    f"  ⏹ Early stopping after {self.patience} evals "
                    f"without improvement. Best: {self.best:.4f}",
                )
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT
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
        if f.startswith(base_name + "_") and f.endswith(".pt")
        and f[len(base_name) + 1:-3].isdigit()
    ])
    for old in versions[:-keep]:
        try:
            os.remove(os.path.join(ckpt_dir, old))
        except OSError:
            pass


def load_ckpt(path, model, opt, sched):
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    try:
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
    except Exception:
        pass
    step = ckpt.get("step", 0)
    _log(f"[green]✔ Resumed from step {step:,}[/]", f"✔ Resumed from step {step:,}")
    return step


# ══════════════════════════════════════════════════════════════════════════════
# 8. DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

_tl_hist: list[float] = []
_vl_hist: list[float] = []


def _log(rich_msg, plain_msg):
    if HAS_RICH:
        console.print(rich_msg)
    else:
        print(plain_msg)


def print_dashboard(step, total, tl, tp, vl, vp, lr, t0, step0, model_id,
                    best_val, early_stop: EarlyStopping):
    done = max(step - step0, 1)
    rate = done / max(time.time() - t0, 1e-3)
    eta  = (total - step) / rate if rate else 0

    _tl_hist.append(tl)
    _vl_hist.append(vl)

    def ft(s):
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    if HAS_RICH:
        t = Table(box=box.ROUNDED, header_style="bold magenta")
        t.add_column("Metric",     style="cyan",   width=16)
        t.add_column("Train",      style="green",  width=12)
        t.add_column("Validation", style="yellow", width=12)
        t.add_row("Loss",       f"{tl:.4f}", f"{vl:.4f}")
        t.add_row("Perplexity", f"{tp:.2f}", f"{vp:.2f}")
        t.add_row("Best Val",   "—",         f"{best_val:.4f}")

        es_str = (f"[red]{early_stop.wait}/{early_stop.patience}[/]"
                  if early_stop.patience > 0 else "[dim]off[/]")
        t.add_row("EarlyStop",  es_str, "")

        s = Table(box=box.ROUNDED, header_style="bold magenta")
        s.add_column("Info",  style="cyan",  width=16)
        s.add_column("Value", style="white", width=14)
        s.add_row("Step",     f"{step}/{total}")
        s.add_row("Progress", f"{100*step/total:.1f}%")
        s.add_row("LR",       f"{lr:.2e}")
        s.add_row("ETA",      ft(eta))
        s.add_row("Speed",    f"{rate:.2f} s/s" if rate > 1 else f"{1/rate:.1f}s/step")

        bars = "▁▂▃▄▅▆▇█"

        def spark(h):
            if len(h) < 2:
                return ""
            mn, mx = min(h), max(h)
            rng = max(mx - mn, 1e-6)
            return "".join(bars[int((v - mn) / rng * 7)] for v in h[-25:])

        console.print(Panel(
            Columns([t, s]),
            title=f"[bold white]🤖 {model_id.split('/')[-1]}[/]",
            subtitle=f"[dim]train: {spark(_tl_hist)} val: {spark(_vl_hist)}[/]",
            border_style="bright_blue",
        ))
    else:
        w = 32
        f = int(w * step / total)
        es_note = (f"  early_stop {early_stop.wait}/{early_stop.patience}"
                   if early_stop.patience else "")
        print(
            f"\n[{'█'*f}{'░'*(w-f)}] {100*step/total:.1f}% step {step}/{total}\n"
            f"  train loss {tl:.4f} ppl {tp:.2f}\n"
            f"  val loss   {vl:.4f} ppl {vp:.2f}  best {best_val:.4f}\n"
            f"  lr {lr:.2e}  ETA {ft(eta)}{es_note}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args      = parse_args()
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16  = device == "cuda" and not args.no_fp16
    model_id  = MODELS[args.model]
    use_lora  = HAS_PEFT and not args.no_lora
    use_qlora = HAS_BNB  and not args.no_qlora and device == "cuda"
    formatter = get_formatter(args.model)

    if HAS_RICH:
        console.print(Panel(
            f"[cyan]Model   :[/] {model_id}\n"
            f"[cyan]Device  :[/] {device}\n"
            f"[cyan]QLoRA   :[/] {'[green]4-bit NF4[/]' if use_qlora else '[yellow]disabled[/]'}\n"
            f"[cyan]LoRA    :[/] {'[green]r='+str(args.lora_r)+' rsLoRA[/]' if use_lora else '[yellow]disabled[/]'}\n"
            f"[cyan]DPO     :[/] {'[green]enabled[/]' if args.dpo or args.dpo_only else '[dim]disabled[/]'}\n"
            f"[cyan]EarlyStop:[/] {'[green]patience='+str(args.early_stop)+'[/]' if args.early_stop else '[dim]disabled[/]'}\n"
            f"[cyan]Steps   :[/] {args.steps:,}  "
            f"[cyan]Eff.batch:[/] {args.batch*args.grad_accum}",
            title="[bold white]🚀 MiAi Trainer[/]",
            border_style="bright_blue",
        ))
    else:
        print(f"\n{'═'*60}")
        print(f"  Model      : {model_id}")
        print(f"  Device     : {device}")
        print(f"  QLoRA      : {use_qlora}")
        print(f"  LoRA r     : {args.lora_r}")
        print(f"  DPO        : {args.dpo or args.dpo_only}")
        print(f"  Early stop : {args.early_stop if args.early_stop else 'off'}")
        print(f"  Steps      : {args.steps:,}  Eff.batch: {args.batch*args.grad_accum}")
        print(f"{'═'*60}\n")

    model, tokenizer = load_model_and_tokenizer(model_id, args, device)

    if args.dpo_only:
        if use_lora:
            model = apply_lora(model, args)
        run_dpo(model, tokenizer, args.save_path, args)
        return

    # ── SFT data ──────────────────────────────────────────────────────────
    examples  = build_sft_examples(args.max_examples, args.max_turns, formatter)
    split     = max(1, int(len(examples) * args.val_split))
    val_ex, train_ex = examples[:split], examples[split:]

    _log("[bold cyan]Tokenising…[/]", "Tokenising…")
    train_ds  = SFTDataset(train_ex, tokenizer, args.block_size, args.model)
    val_ds    = SFTDataset(val_ex,   tokenizer, args.block_size, args.model)
    collate   = make_collate(tokenizer.pad_token_id)
    train_dl  = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                           collate_fn=collate, drop_last=True,
                           pin_memory=(device == "cuda"))
    val_dl    = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                           collate_fn=collate)

    if use_lora:
        model = apply_lora(model, args)
    if device == "cpu":
        model = model.to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────
    try:
        from bitsandbytes.optim import PagedAdamW32bit
        optimizer = PagedAdamW32bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01,
        )
        _log("[green]  Using PagedAdamW[/]", "  Using PagedAdamW")
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.95),
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=args.steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    start = load_ckpt(args.checkpoint, model, optimizer, scheduler)

    # ── Early stopping setup ──────────────────────────────────────────────
    early_stop = EarlyStopping(patience=args.early_stop)

    _log(f"[bold white]SFT: step {start:,} → {args.steps:,}[/]\n",
         f"SFT: step {start:,} → {args.steps:,}\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0, step = time.time(), start
    accum_loss = 0.0
    best_val   = float("inf")
    train_it   = iter(train_dl)
    log_file   = open(args.log_file, "a")
    stopped_early = False

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
                try:
                    batch = next(train_it)
                except StopIteration:
                    train_it = iter(train_dl)
                    batch    = next(train_it)

                with torch.autocast(device_type=device, dtype=torch.float16,
                                    enabled=use_fp16 and device == "cuda"):
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )

                if args.label_smooth > 0 and not torch.isnan(out.loss):
                    B, T, V = out.logits.shape
                    lp  = F.log_softmax(out.logits, dim=-1)
                    sl  = -lp.mean(-1)
                    valid = batch["labels"].to(device) != -100
                    sl  = (sl * valid).sum() / valid.sum().clamp(1)
                    loss = ((1 - args.label_smooth) * out.loss +
                            args.label_smooth * sl) / args.grad_accum
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
            if HAS_RICH:
                progress.advance(task)

            # ── Eval & early stopping ─────────────────────────────────────
            if step % args.eval_every == 0 or step == args.steps:
                vl, vp   = run_eval(model, val_dl, args.eval_iters, device, use_fp16)
                tl       = accum_loss
                tp       = math.exp(min(tl, 20))
                lr_now   = scheduler.get_last_lr()[0]

                if HAS_RICH:
                    progress.stop()
                print_dashboard(step, args.steps, tl, tp, vl, vp,
                                lr_now, t0, start, model_id, best_val, early_stop)
                if HAS_RICH:
                    progress.start()

                log_file.write(json.dumps({
                    "step": step, "train_loss": tl, "val_loss": vl,
                    "val_ppl": vp, "lr": lr_now, "time": time.time() - t0,
                }) + "\n")
                log_file.flush()

                if vl < best_val:
                    best_val = vl
                    model.save_pretrained(args.save_path + "-best")
                    tokenizer.save_pretrained(args.save_path + "-best")
                    _log(f"[bold green]⭐ Best val {vl:.4f} → {args.save_path}-best/[/]",
                         f"⭐ Best val {vl:.4f}")

                # Check early stopping
                if early_stop(vl):
                    stopped_early = True
                    break

                accum_loss = 0.0

            if step % args.save_every == 0:
                save_ckpt(args.checkpoint, step, model, optimizer, scheduler,
                          model_id, args.keep_ckpts)

            if stopped_early:
                break

    finally:
        if HAS_RICH:
            progress.stop()
        log_file.close()

    # ── Final save ────────────────────────────────────────────────────────
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    save_ckpt(args.checkpoint, step, model, optimizer, scheduler,
              model_id, args.keep_ckpts)

    if stopped_early:
        _log(f"[yellow]  Training ended early at step {step:,}. "
             f"Best model already saved to {args.save_path}-best/[/]",
             f"  Stopped early at step {step:,}. Best → {args.save_path}-best/")
    else:
        _log(f"[green]✔ SFT saved → {args.save_path}/[/]",
             f"✔ SFT saved → {args.save_path}/")

    if args.dpo:
        run_dpo(model, tokenizer, args.save_path, args)

    elapsed = time.time() - t0
    m, s    = divmod(int(elapsed), 60)
    h, m    = divmod(m, 60)
    msg = (
        f"Total time  : {h:02d}:{m:02d}:{s:02d}\n"
        f"Last model  → ./{args.save_path}/\n"
        f"Best model  → ./{args.save_path}-best/\n"
        + (f"DPO model   → ./{args.save_path}-dpo/\n" if args.dpo else "")
        + f"Run         → python3 chat.py"
    )
    if HAS_RICH:
        console.print(Panel(msg, title="[bold green]✅ Complete[/]", border_style="green"))
    else:
        print(f"\n✅ {msg}")


if __name__ == "__main__":
    main()
