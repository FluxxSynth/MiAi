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
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


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
SYSTEM_MSG = (
    "You are a helpful, honest, and harmless AI assistant. "
    "You give clear, accurate, and thoughtful responses."
)

def detect_format(model_path: str) -> str:
    """Detect model family from path/config."""
    path_lower = model_path.lower()
    if "llama" in path_lower:   return "llama3"
    if "phi"   in path_lower:   return "phi3"
    if "gemma" in path_lower:   return "gemma"
    return "mistral"  # default

def build_prompt(turns: list[dict], user_msg: str,
                 fmt: str, kb_ctx: str = "") -> str:
    """
    Build the correctly formatted prompt for the detected model family.
    Includes conversation history and optional RAG context.
    """
    system = SYSTEM_MSG
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
            except (json.JSONDecodeError, OSError) as e:
                print(f"⚠  Could not load history: {e}")

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
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0a0a14;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
header{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:14px 20px;
       display:flex;align-items:center;gap:12px;border-bottom:1px solid #2d2d5e;
       box-shadow:0 2px 20px rgba(0,0,0,.5)}
header h1{font-size:1.15rem;font-weight:700;color:#a78bfa}
.tag{font-size:.72rem;background:#252550;color:#818cf8;padding:3px 10px;
     border-radius:20px;font-weight:500}
#persona-select{margin-left:auto;background:#1e1e3f;color:#c4b5fd;
                border:1px solid #4c4c8f;border-radius:8px;padding:6px 10px;
                font-size:.82rem;cursor:pointer;outline:none}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px;
      scroll-behavior:smooth}
.row{display:flex;align-items:flex-end;gap:10px;animation:fadeIn .2s ease}
.row.user{flex-direction:row-reverse}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.av{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:1.1rem;flex-shrink:0}
.av.bot{background:linear-gradient(135deg,#4f46e5,#7c3aed)}
.av.user{background:linear-gradient(135deg,#0284c7,#0ea5e9)}
.bubble{max-width:72%;padding:12px 16px;border-radius:18px;font-size:.9rem;
        line-height:1.6;position:relative}
.bubble.bot{background:#1a1a35;border:1px solid #2d2d5e;border-bottom-left-radius:4px}
.bubble.user{background:linear-gradient(135deg,#4338ca,#6d28d9);color:#fff;
             border-bottom-right-radius:4px}
.ts{font-size:.65rem;color:#6b7280;margin-top:6px}
.bubble.user .ts{color:rgba(255,255,255,.45)}
.typing{display:flex;gap:5px;padding:10px 14px;align-items:center}
.typing span{width:8px;height:8px;background:#6366f1;border-radius:50%;
             animation:bounce 1.1s infinite}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-7px)}}
#bar{padding:8px 20px;background:#111126;border-top:1px solid #1e1e3f;
     display:flex;gap:8px;flex-wrap:wrap}
.btn{background:#1e1e3f;border:1px solid #3d3d6b;color:#a78bfa;border-radius:8px;
     padding:5px 13px;font-size:.78rem;cursor:pointer;transition:background .15s}
.btn:hover{background:#2d2d5e}
#status{font-size:.75rem;color:#4b5563;margin-left:auto;align-self:center}
#footer{padding:12px 20px;background:#111126;border-top:1px solid #1e1e3f;
        display:flex;gap:10px}
#msg{flex:1;background:#1a1a35;border:1px solid #3d3d6b;color:#e2e8f0;
     border-radius:12px;padding:11px 16px;font-size:.9rem;resize:none;
     outline:none;max-height:130px;transition:border-color .2s}
#msg:focus{border-color:#6366f1}
#send{background:linear-gradient(135deg,#4338ca,#7c3aed);color:#fff;border:none;
      border-radius:12px;padding:11px 22px;cursor:pointer;font-weight:600;
      font-size:.9rem;transition:opacity .2s;white-space:nowrap}
#send:hover{opacity:.85}
#send:disabled{opacity:.35;cursor:not-allowed}
pre{background:#0d0d20;border:1px solid #2d2d5e;border-radius:8px;
    padding:12px;overflow-x:auto;margin:6px 0;font-size:.82em}
code{background:#1a1a35;border-radius:4px;padding:1px 5px;font-size:.85em}
</style>
</head>
<body>
<header>
  <div>🤖</div>
  <h1>AI Chatbot</h1>
  <span class="tag" id="mtag">loading…</span>
  <select id="persona-select" onchange="setPersona(this.value)">
    <option value="assistant">🤖 Assistant</option>
    <option value="tutor">📚 Tutor</option>
    <option value="coder">💻 Coder</option>
    <option value="creative">✨ Creative</option>
  </select>
</header>
<div id="chat"></div>
<div id="bar">
  <button class="btn" onclick="clearChat()">🗑 Clear</button>
  <button class="btn" onclick="saveHistory()">💾 Save</button>
  <button class="btn" onclick="retryLast()">↩ Retry</button>
  <button class="btn" onclick="showHelp()">❓ Help</button>
  <span id="status"></span>
</div>
<div id="footer">
  <textarea id="msg" rows="1" placeholder="Message… (Enter = send, Shift+Enter = newline)"
            onkeydown="onKey(event)"></textarea>
  <button id="send" onclick="send()">Send ↑</button>
</div>
<script>
let lastUser="";
async function info(){
  const r=await fetch("/api/info");const d=await r.json();
  document.getElementById("mtag").textContent=d.model;
}
function ts(){return new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})}
function md(t){
  return t.replace(/```([\\w]*)\\n?([\\s\\S]*?)```/g,"<pre><code>$2</code></pre>")
          .replace(/`([^`]+)`/g,"<code>$1</code>")
          .replace(/\\*\\*(.*?)\\*\\*/g,"<b>$1</b>")
          .replace(/\\*(.*?)\\*/g,"<i>$1</i>")
          .replace(/\\n/g,"<br>");
}
function addBubble(role,text){
  const chat=document.getElementById("chat");
  const row=document.createElement("div");row.className=`row ${role}`;
  const av=document.createElement("div");av.className=`av ${role}`;
  av.textContent=role==="bot"?"🤖":"🧑";
  const bub=document.createElement("div");bub.className=`bubble ${role}`;
  bub.innerHTML=md(text)+`<div class="ts">${ts()}</div>`;
  row.appendChild(av);row.appendChild(bub);
  chat.appendChild(row);chat.scrollTop=chat.scrollHeight;
}
function addTyping(){
  const chat=document.getElementById("chat");
  const row=document.createElement("div");row.id="tr";row.className="row bot";
  row.innerHTML=`<div class="av bot">🤖</div>
    <div class="bubble bot"><div class="typing">
      <span></span><span></span><span></span></div></div>`;
  chat.appendChild(row);chat.scrollTop=chat.scrollHeight;
}
function rmTyping(){const t=document.getElementById("tr");if(t)t.remove();}
async function send(){
  const inp=document.getElementById("msg");
  const txt=inp.value.trim();if(!txt)return;
  lastUser=txt;inp.value="";inp.style.height="auto";
  document.getElementById("send").disabled=true;
  document.getElementById("status").textContent="Thinking…";
  addBubble("user",txt);addTyping();
  try{
    const r=await fetch("/api/chat",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message:txt})});
    const d=await r.json();rmTyping();
    addBubble("bot",d.response);
    document.getElementById("status").textContent=
      `${d.tokens} tok · ${d.ms}ms`;
  }catch(e){rmTyping();addBubble("bot","⚠ Server error.");}
  document.getElementById("send").disabled=false;
  document.getElementById("msg").focus();
}
async function clearChat(){
  await fetch("/api/reset",{method:"POST"});
  document.getElementById("chat").innerHTML="";
  document.getElementById("status").textContent="Cleared";
}
async function saveHistory(){
  await fetch("/api/save",{method:"POST"});
  document.getElementById("status").textContent="Saved ✔";
}
async function retryLast(){
  if(!lastUser)return;
  await fetch("/api/retry",{method:"POST"});
  const chat=document.getElementById("chat");
  const bots=chat.querySelectorAll(".row.bot");
  if(bots.length)bots[bots.length-1].remove();
  addTyping();
  const r=await fetch("/api/chat",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:lastUser,retry:true})});
  const d=await r.json();rmTyping();addBubble("bot",d.response);
}
async function setPersona(p){
  await fetch("/api/persona",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({persona:p})});
  document.getElementById("status").textContent="Persona: "+p;
}
function showHelp(){
  addBubble("bot",
    "**Commands:**\n" +
    "• Clear — wipe conversation\n" +
    "• Save — save history to disk\n" +
    "• Retry — regenerate last response\n" +
    "• Persona — change bot personality\n\n" +
    "**Tips:**\n" +
    "• Shift+Enter for newlines\n" +
    "• The bot remembers your conversation history");
}
function onKey(e){
  if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}
  const t=e.target;t.style.height="auto";
  t.style.height=Math.min(t.scrollHeight,130)+"px";
}
info();
</script>
</body>
</html>
"""

def create_app(model, tokenizer, conv: Conversation,
               kb: KnowledgeBase, args, device, fmt):
    app = Flask(__name__)

    @app.route("/")
    def index(): return HTML

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

        prompt   = build_prompt(conv.recent(), msg, fmt, kb_ctx)
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
                    prompt   = build_prompt(conv.recent(), last_user, fmt)
                    response = generate(model, tokenizer, prompt, args, device, fmt)
                    conv.add_bot(response)
                    print(f"Bot: {textwrap.fill(response,80,subsequent_indent='     ')}\n")
            elif cmd=="/beams" and len(parts)>1:
                args.beams=int(parts[1]); print(f"  ✔ beams → {args.beams}\n")
            continue

        last_user = raw
        conv.add_user(raw)
        kb_ctx   = kb.retrieve(raw) if kb.loaded else ""
        prompt   = build_prompt(conv.recent(), raw, fmt, kb_ctx)
        response = generate(model, tokenizer, prompt, args, device, fmt)
        conv.add_bot(response)
        print(f"Bot: {textwrap.fill(response,80,subsequent_indent='     ')}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.model):
        print(f"\n❌  No model at '{args.model}'. Run train.py first.\n")
        sys.exit(1)

    model, tokenizer = load_model(args.model, args, device)
    fmt  = detect_format(args.model)
    kb   = KnowledgeBase(args.kb) if args.kb else KnowledgeBase(None)
    conv = Conversation(args.history_file)

    if not args.no_web:
        if HAS_FLASK:
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
            print("⚠  Flask not installed. Install with: pip install flask")
            print("   Falling back to terminal mode.\n")
            terminal_loop(model, tokenizer, conv, kb, args, device, fmt)
    else:
        terminal_loop(model, tokenizer, conv, kb, args, device, fmt)


if __name__ == "__main__":
    main()
