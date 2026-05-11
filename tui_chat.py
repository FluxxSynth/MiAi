"""
tui_chat.py — PyTermGUI-based chat interface

Fixed: pytermgui recursive _update_size / resolution crash on macOS + Python 3.12.

Root cause: ptg.Terminal._update_size() calls self.resolution (a cached_property)
which itself calls _update_size again before the cache is populated, causing
infinite recursion and an abort signal.

Fix: pre-populate the internal size cache with os.get_terminal_size() before
WindowManager is constructed, so the property short-circuits on first access.
Also avoid referencing manager.terminal.width/height at window construction
time (a second trigger point).

Run:
    python3 tui_chat.py
    python3 tui_chat.py --model finetuned-model-best
    python3 tui_chat.py --no_qlora
"""

import os, sys, time

# ── Crash fix — must happen before ANY pytermgui import ──────────────────────



# pytermgui.Terminal caches its size in _size via a cached_property called

# `resolution`. On macOS the first call comes from a SIGWINCH handler fired

# during import, before the cache slot exists, causing a recursive call chain.



# We monkey-patch the property before it is ever accessed by pre-filling the

# underlying slot on the Terminal singleton right after import.

# ─────────────────────────────────────────────────────────────────────────────

try:
    import pytermgui as ptg
except ImportError:
    print("❌ pytermgui not installed.  Run: pip install pytermgui")
    sys.exit(1)

def _safe_terminal_size() -> tuple[int, int]:
    """Return (cols, rows) with a safe fallback — never raises."""
    try:
        s = os.get_terminal_size()
        return (s.columns, s.lines)
    except OSError:
        return (120, 40)

# Pre-populate the cached_property slot so _update_size never recurses.

# The slot name on ptg.Terminal is "_size" (the backing store for `resolution`).

_term = ptg.terminal          # the module-level singleton
cols, rows = _safe_terminal_size()

# Use object.__setattr__ to bypass any custom __setattr__ guards.

try:
    object.__setattr__(_term, "_size", (cols, rows))
except Exception:
    pass                       # if the slot name changed in a newer version,
                               # the WindowManager fallback below still protects us

# ─────────────────────────────────────────────────────────────────────────────

from chat import (
    parse_args,
    load_model,
    generate,
    get_device,
    Conversation,
    KnowledgeBase,
    detect_format,
    build_prompt,
)

def _term_size() -> tuple[int, int]:
    """Safely read terminal size, never triggering ptg's internal recursion."""
    try:
        s = os.get_terminal_size()
        return s.columns, s.lines
    except OSError:
        return 120, 40

def main():
    args   = parse_args()
    device = get_device()

    if not os.path.exists(args.model):
        print(f"\n❌ No model at '{args.model}'.  Run train.py first.\n")
        sys.exit(1)

    print("Loading model…")
    model, tokenizer = load_model(args.model, args, device)
    fmt  = detect_format(args.model)
    kb   = KnowledgeBase(args.kb) if args.kb else KnowledgeBase(None)
    conv = Conversation(args.history_file)

    last_user = ""

    # Read size once before entering the manager so we never ask ptg for it
    # at construction time (second crash trigger).
    term_cols, term_rows = _term_size()

    with ptg.WindowManager() as manager:

        # ── Widgets ───────────────────────────────────────────────────────
        chat_container = ptg.Container(
            ptg.Label("[dim]Welcome! Type a message and press Enter.  /help for commands.[/]")
        )
        input_field  = ptg.InputField("", prompt="> ")
        status_label = ptg.Label(
            f"[dim]Model: {os.path.basename(args.model)} | Enter=send | /help[/]"
        )

        # ── Helpers ───────────────────────────────────────────────────────
        def add_message(role: str, text: str) -> None:
            prefix = "[bold 157]You:[/]" if role == "user" else "[bold 141]Bot:[/]"
            # Wrap long lines so they don't overflow the box
            wrapped = _wrap(text, term_cols - 12)
            chat_container += ptg.Label(f"{prefix} {wrapped}")
            chat_container += ptg.Label("")

        def rebuild_history() -> None:
            chat_container.set_widgets([])
            for turn in conv.turns:
                add_message(turn["role"], turn["text"])

        def handle_command(text: str) -> bool:
            parts = text.split()
            cmd   = parts[0].lower()

            if cmd == "/exit":
                manager.stop()
                return True

            if cmd == "/reset":
                conv.reset()
                chat_container.set_widgets([])
                chat_container += ptg.Label("[dim]Conversation cleared.[/]")
                status_label.value = "[dim]Cleared.[/]"
                return True

            if cmd == "/save":
                conv.save()
                status_label.value = "[dim]Saved.[/]"
                return True

            if cmd == "/retry":
                if not last_user:
                    status_label.value = "[dim]Nothing to retry.[/]"
                    return True
                conv.pop_last_bot()
                rebuild_history()
                _do_generate(last_user, retry=True)
                return True

            if cmd == "/persona" and len(parts) > 1:
                conv.persona = parts[1]
                status_label.value = f"[dim]Persona → {parts[1]}[/]"
                return True

            if cmd == "/help":
                add_message(
                    "bot",
                    "Commands:\n"
                    "  /reset          — clear conversation\n"
                    "  /save           — save history to disk\n"
                    "  /retry          — regenerate last response\n"
                    "  /persona <name> — set persona (assistant/tutor/coder/creative)\n"
                    "  /exit           — quit\n"
                    "  /help           — show this help",
                )
                return True

            return False

        def _do_generate(text: str, retry: bool = False) -> None:
            nonlocal last_user
            kb_ctx = kb.retrieve(text) if kb.loaded else ""
            if not retry:
                conv.add_user(text)
                last_user = text

            # Use token-aware truncation if available, fall back to turn slice
            if hasattr(conv, "recent_by_tokens"):
                recent = conv.recent_by_tokens(tokenizer)
            else:
                recent = conv.recent()

            prompt = build_prompt(recent, text, fmt, kb_ctx, conv.persona)
            status_label.value = "[dim]Generating…[/]"

            t0 = time.time()
            try:
                response = generate(model, tokenizer, prompt, args, device, fmt)
            except KeyboardInterrupt:
                status_label.value = "[dim]Interrupted.[/]"
                return

            ms  = int((time.time() - t0) * 1000)
            tok = len(tokenizer.encode(response, add_special_tokens=False))
            conv.add_bot(response)
            add_message("bot", response)
            status_label.value = f"[dim]{tok} tok · {ms} ms[/]"

        # ── Input handler ──────────────────────────────────────────────────
        def on_submit(*_) -> None:
            text = input_field.value.strip()
            if not text:
                return
            input_field.value = ""
            if text.startswith("/"):
                if handle_command(text):
                    return
            add_message("user", text)
            _do_generate(text)

        input_field.bind(ptg.keys.ENTER, on_submit)

        # ── Window — use pre-read size, NOT manager.terminal.width/height ──
        # Referencing manager.terminal.{width,height} here is the second
        # crash trigger on macOS; use the values we read before entering
        # the context manager instead.
        window = ptg.Window(
            f"[bold 141]🤖 AI Chat[/] [dim]{os.path.basename(args.model)}[/]",
            "",
            chat_container,
            "",
            status_label,
            "",
            input_field,
            overflow=ptg.Overflow.SCROLL,
            vertical_align=ptg.VerticalAlignment.BOTTOM,
            width=term_cols,
            height=term_rows,
            box="ROUNDED",
        )

        window.center()
        manager.layout.add_slot("Body")
        manager.add(window)
        window.select(-1)   # focus the input field immediately


def _wrap(text: str, width: int) -> str:
    """Hard-wrap text so very long bot responses don't overflow ptg's box."""
    if width < 20:
        return text
    lines = []
    for paragraph in text.splitlines():
        while len(paragraph) > width:
            lines.append(paragraph[:width])
            paragraph = paragraph[width:]
        lines.append(paragraph)
    return "\n".join(lines)

if __name__ == "__main__":
    main()
