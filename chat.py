"""
chat.py — Mistral-7B / LLaMA-3 chatbot
========================================
Works with any model saved by the new train.py.
Falls back gracefully to CPU if no GPU available.

Run:
  python3 chat.py                              # web UI on http://localhost:5000
  python3 chat.py --model finetuned-model-best # best checkpoint
  python3 chat.py --model finetuned-model-dpo  # DPO-aligned model
  python3 chat.py --no_web                     # terminal only
  python3 chat.py --kb ./my_docs/              # add knowledge base
"""

import os, sys, json, time, argparse, textwrap, threading, re
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import bitsandbytes
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    from flask import Flask, request, jsonify, render_template
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("⚠  pip install flask  for web UI")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  ARGS
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",        default="finetuned-model")
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--top_k",        type=int,   default=50)
    p.add_argument("--top_p",        type=float, default=0.92)
    p.add_argument("--max_tokens",   type=int,   default=512)
    p.add_argument("--beams",        type=int,   default=1,
                   help="Beam search width (1=sampling, 3=coherent)")
    p.add_argument("--rep_penalty",  type=float, default=1.15)
    p.add_argument("--kb",           type=str,   default=None,
                   help="Knowledge base folder (.txt files)")
    p.add_argument("--history_file", default="chat_history.json")
    p.add_argument("--port",         type=int,   default=5000)
    p.add_argument("--no_web",       action="store_true")
    p.add_argument("--no_qlora",     action="store_true",
                   help="Disable 4-bit loading (use if model wasn't QLoRA trained)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_model(model_path: str, args, device: str):
    """
    Auto-detects model family from config and loads with appropriate settings.
    Applies 4-bit quantization for inference if bitsandbytes is available.
    Automatically detects and loads PEFT/LoRA adapters saved by train.py.
    """
    print(f"Loading model from '{model_path}'…", end=" ", flush=True)

    # Detect PEFT adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_adapter = os.path.exists(adapter_config_path)

    if is_adapter and not HAS_PEFT:
        print("\n❌  PEFT adapter found but peft is not installed.  pip install peft")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation

    # 4-bit inference — same memory savings as training
    if HAS_BNB and not args.no_qlora and device == "cuda":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit           = True,
            bnb_4bit_quant_type    = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
        )
    else:
        bnb_cfg = None

    if is_adapter:
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_id = adapter_cfg.get("base_model_name_or_path")
        if not base_model_id:
            print("\n❌  adapter_config.json missing base_model_name_or_path")
            sys.exit(1)
        print(f"\n  PEFT adapter detected — loading base model {base_model_id}…",
              end=" ", flush=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config = bnb_cfg,
            torch_dtype         = torch.float16 if (device=="cuda" and not bnb_cfg) else torch.float32,
            device_map          = "auto" if device == "cuda" else None,
            trust_remote_code   = True,
        )
        if device == "cpu":
            base_model = base_model.to(device)

        model = PeftModel.from_pretrained(base_model, model_path)
        print("merging adapter…", end=" ", flush=True)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config = bnb_cfg,
            torch_dtype         = torch.float16 if (device=="cuda" and not bnb_cfg) else torch.float32,
            device_map          = "auto" if device == "cuda" else None,
            trust_remote_code   = True,
        )

        if device == "cpu":
            model = model.to(device)

    model.eval()
    print("done.")

    n = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n:,}  |  Device: {device}"
          + ("  |  4-bit" if bnb_cfg else "")
          + ("  |  LoRA merged" if is_adapter else ""))

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROMPT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════
PERSONAS = {
    "assistant": (
        "You are a helpful, honest, and harmless AI assistant. "
        "You give clear, accurate, and thoughtful responses."
    ),
    "tutor": (
        "You are a knowledgeable and patient tutor. You explain concepts "
        "clearly, break down complex topics, and provide examples to aid understanding."
    ),
    "coder": (
        "You are an expert programmer. You write clean, well-documented code "
        "and explain technical concepts clearly with practical examples."
    ),
    "creative": (
        "You are a creative writing assistant with a flair for imaginative "
        "storytelling. You help craft engaging narratives and think outside the box."
    ),
}

SYSTEM_MSG = PERSONAS["assistant"]

def detect_format(model_path: str) -> str:
    """Detect model family from path/config."""
    path_lower = model_path.lower()
    if "llama" in path_lower:   return "llama3"
    if "phi"   in path_lower:   return "phi3"
    if "gemma" in path_lower:   return "gemma"
    return "mistral"  # default

def build_prompt(turns: list[dict], user_msg: str,
                 fmt: str, kb_ctx: str = "", persona: str = "assistant") -> str:
    """
    Build the correctly formatted prompt for the detected model family.
    Includes conversation history and optional RAG context.
    Uses the selected persona system message if available.
    """
    system = PERSONAS.get(persona, PERSONAS["assistant"])
    if kb_ctx:
        system += f"\n\nRelevant context:\n{kb_ctx}"

    history = [(t["text"], turns[i+1]["text"])
               for i, t in enumerate(turns[:-1])
               if t["role"] == "user" and i+1 < len(turns) and turns[i+1]["role"] == "bot"]

    # Add current message
    all_turns = history + [(user_msg, None)]

    if fmt == "llama3":
        out = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
               f"\n{system}<|eot_id|>")
        for u, a in all_turns:
            out += f"<|start_header_id|>user<|end_header_id|>\n{u}<|eot_id|>"
            out += f"<|start_header_id|>assistant<|end_header_id|>"
            if a:
                out += f"\n{a}<|eot_id|>"
        return out

    elif fmt == "phi3":
        out = f"<|system|>\n{system}<|end|>\n"
        for u, a in all_turns:
            out += f"<|user|>\n{u}<|end|>\n<|assistant|>"
            if a: out += f"\n{a}<|end|>\n"
        return out

    elif fmt == "gemma":
        out = ""
        for u, a in all_turns:
            out += f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model"
            if a: out += f"\n{a}<end_of_turn>\n"
        return out

    else:  # mistral
        out = ""
        for i, (u, a) in enumerate(all_turns):
            if i == 0:
                out += f"<s>[INST] {system}\n\n{u} [/INST]"
            else:
                out += f"[INST] {u} [/INST]"
            if a: out += f" {a} </s>"
        return out


def get_stop_strings(fmt: str) -> list[str]:
    stops = {
        "llama3":  ["<|eot_id|>", "<|start_header_id|>user"],
        "phi3":    ["<|end|>", "<|user|>"],
        "gemma":   ["<end_of_turn>", "<start_of_turn>user"],
        "mistral": ["</s>", "[INST]"],
    }
    return stops.get(fmt, ["</s>"])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  KNOWLEDGE BASE (RAG)
# ══════════════════════════════════════════════════════════════════════════════
class KnowledgeBase:
    def __init__(self, folder: str, chunk_size: int = 400):
        self.chunks: list[str] = []
        self.loaded = False
        if not folder or not os.path.exists(folder):
            return
        files = list(Path(folder).rglob("*.txt"))
        print(f"📚  Indexing {len(files)} knowledge base files…")
        for f in files:
            text  = f.read_text(encoding="utf-8", errors="ignore")
            words = text.split()
            for i in range(0, len(words), chunk_size // 2):
                chunk = " ".join(words[i: i + chunk_size])
                if len(chunk) > 50:
                    self.chunks.append(chunk)
        self.loaded = bool(self.chunks)
        print(f"✔  {len(self.chunks):,} chunks indexed.")

    def retrieve(self, query: str, top_k: int = 2) -> str:
        if not self.loaded: return ""
        qw = set(re.findall(r"\w+", query.lower()))
        scores = []
        for chunk in self.chunks:
            cw = re.findall(r"\w+", chunk.lower())
            scores.append(sum(1 for w in cw if w in qw) / max(len(cw), 1))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chunks = [self.chunks[i] for i in top[:top_k] if scores[i] > 0]
        return "\n\n".join(f"[Ref {i+1}]: {c}" for i, c in enumerate(chunks))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONVERSATION
# ══════════════════════════════════════════════════════════════════════════════
class Conversation:
    def __init__(self, history_file: str):
        self.turns: list[dict] = []
        self.persona = "assistant"
        self.history_file = history_file
        self._load()

    def reset(self):
        self.turns.clear()

    def add_user(self, text: str):
        self.turns.append({"role": "user", "text": text,
                           "ts": datetime.now().strftime("%H:%M")})

    def add_bot(self, text: str):
        self.turns.append({"role": "bot", "text": text,
                           "ts": datetime.now().strftime("%H:%M")})

    def pop_last_bot(self):
        for i in range(len(self.turns)-1, -1, -1):
            if self.turns[i]["role"] == "bot":
                self.turns.pop(i); return

    def save(self):
        with open(self.history_file, "w") as f:
            json.dump({"turns": self.turns}, f, indent=2)

    def _load(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self.turns = data.get("turns", [])
                print(f"✔  Loaded {len(self.turns)} turns from history.")
            except: pass

    # Keep only recent turns that fit in context
    def recent(self, max_turns: int = 10) -> list[dict]:
        return self.turns[-max_turns*2:]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate(model, tokenizer, prompt: str, args, device: str,
             fmt: str) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    stop_strs = get_stop_strings(fmt)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens       = args.max_tokens,
            do_sample            = args.beams == 1,
            temperature          = args.temperature if args.beams==1 else 1.0,
            top_k                = args.top_k       if args.beams==1 else 0,
            top_p                = args.top_p       if args.beams==1 else 1.0,
            num_beams            = args.beams,
            repetition_penalty   = args.rep_penalty,
            no_repeat_ngram_size = 4,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.pad_token_id,
            use_cache            = True,
        )

    new_ids  = output[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Clean up stop strings
    for stop in stop_strs:
        if stop in response:
            response = response[:response.index(stop)].strip()

    return response or "(no response)"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  WEB UI
# ══════════════════════════════════════════════════════════════════════════════
# HTML template extracted to templates/chat.html

def create_app(model, tokenizer, conv: Conversation,
               kb: KnowledgeBase, args, device, fmt):
    app = Flask(__name__)

    @app.route("/")
    def index(): return render_template("chat.html")

    @app.route("/api/info")
    def info(): return jsonify({"model": os.path.basename(args.model)})

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data  = request.json
        msg   = data.get("message","").strip()
        retry = data.get("retry", False)
        if not msg:
            return jsonify({"response":"","tokens":0,"ms":0})

        kb_ctx = kb.retrieve(msg) if kb.loaded else ""
        if not retry:
            conv.add_user(msg)

        prompt   = build_prompt(conv.recent(), msg, fmt, kb_ctx, conv.persona)
        t0       = time.time()
        response = generate(model, tokenizer, prompt, args, device, fmt)
        ms       = int((time.time()-t0)*1000)
        tokens   = len(tokenizer.encode(response))

        conv.add_bot(response)
        return jsonify({"response":response,"tokens":tokens,"ms":ms})

    @app.route("/api/reset", methods=["POST"])
    def reset(): conv.reset(); return jsonify({"ok":True})

    @app.route("/api/save", methods=["POST"])
    def save(): conv.save(); return jsonify({"ok":True})

    @app.route("/api/retry", methods=["POST"])
    def retry(): conv.pop_last_bot(); return jsonify({"ok":True})

    @app.route("/api/persona", methods=["POST"])
    def persona():
        conv.persona = request.json.get("persona","assistant")
        return jsonify({"ok":True})

    return app


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TERMINAL MODE
# ══════════════════════════════════════════════════════════════════════════════
def terminal_loop(model, tokenizer, conv, kb, args, device, fmt):
    print(f"\n{'═'*55}")
    print("  AI Chatbot — terminal mode")
    print("  Commands: /reset  /save  /retry  /beams N  exit")
    print(f"{'═'*55}\n")

    last_user = ""
    while True:
        try:   raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break

        if not raw: continue
        if raw.lower() in {"exit","quit"}: print("Goodbye!"); break

        if raw.startswith("/"):
            parts = raw.split()
            cmd = parts[0].lower()
            if   cmd=="/reset":  conv.reset(); print("  ✔ Cleared.\n")
            elif cmd=="/save":   conv.save(); print("  ✔ Saved.\n")
            elif cmd=="/retry":
                if last_user:
                    conv.pop_last_bot()
                    prompt   = build_prompt(conv.recent(), last_user, fmt, persona=conv.persona)
                    response = generate(model, tokenizer, prompt, args, device, fmt)
                    conv.add_bot(response)
                    print(f"Bot: {textwrap.fill(response,80,subsequent_indent='     ')}\n")
            elif cmd=="/beams" and len(parts)>1:
                args.beams=int(parts[1]); print(f"  ✔ beams → {args.beams}\n")
            continue

        last_user = raw
        conv.add_user(raw)
        kb_ctx   = kb.retrieve(raw) if kb.loaded else ""
        prompt   = build_prompt(conv.recent(), raw, fmt, kb_ctx, conv.persona)
        response = generate(model, tokenizer, prompt, args, device, fmt)
        conv.add_bot(response)
        print(f"Bot: {textwrap.fill(response,80,subsequent_indent='     ')}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  DEVICE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_device() -> str:
    """Detect available device — cuda if available else cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = get_device()

    if not os.path.exists(args.model):
        print(f"\n❌  No model at '{args.model}'. Run train.py first.\n")
        sys.exit(1)

    model, tokenizer = load_model(args.model, args, device)
    fmt  = detect_format(args.model)
    kb   = KnowledgeBase(args.kb) if args.kb else KnowledgeBase(None)
    conv = Conversation(args.history_file)

    if not args.no_web and HAS_FLASK:
        app = create_app(model, tokenizer, conv, kb, args, device, fmt)
        url = f"http://localhost:{args.port}"
        print(f"\n🌐  Web UI → {url}")
        print("    Open in your browser. Ctrl+C to stop.\n")
        def _open():
            time.sleep(1.5)
            import webbrowser; webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()
        app.run(host="0.0.0.0", port=args.port, debug=False)
    else:
        terminal_loop(model, tokenizer, conv, kb, args, device, fmt)


if __name__ == "__main__":
    main()
