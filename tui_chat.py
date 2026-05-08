"""
tui_chat.py — PyTermGUI-based chat interface
============================================
Reuses chat.py's model loading and generation.

Run:
  pip3 install pytermgui
  python3 tui_chat.py
  python3 tui_chat.py --model finetuned-model-best
  python3 tui_chat.py --no_qlora
"""

import os, sys, time

try:
    import pytermgui as ptg
except ImportError:
    print("❌  pytermgui not installed. Run: pip3 install pytermgui")
    sys.exit(1)

import torch

# Reuse chat.py logic
from chat import (
    parse_args,
    load_model,
    generate,
    Conversation,
    KnowledgeBase,
    detect_format,
    build_prompt,
)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.model):
        print(f"\n❌  No model at '{args.model}'. Run train.py first.\n")
        sys.exit(1)

    print("Loading model...")
    model, tokenizer = load_model(args.model, args, device)
    fmt = detect_format(args.model)
    kb = KnowledgeBase(args.kb) if args.kb else KnowledgeBase(None)
    conv = Conversation(args.history_file)

    # Track last user message for retry
    last_user = ""

    with ptg.WindowManager() as manager:
        # ── Widgets ──────────────────────────────────────────────────────────
        chat_container = ptg.Container(
            ptg.Label(
                "[dim]Welcome! Type a message and press Enter.  /help for commands.[/]"
            )
        )

        input_field = ptg.InputField("", prompt="> ")

        status_label = ptg.Label(
            f"[dim]Model: {os.path.basename(args.model)} | Enter=send | /help[/]"
        )

        # ── Helpers ──────────────────────────────────────────────────────────
        def add_message(role: str, text: str):
            prefix = (
                "[bold 157]You:[/]"
                if role == "user"
                else "[bold 141]Bot:[/]"
            )
            chat_container += ptg.Label(f"{prefix} {text}")
            chat_container += ptg.Label("")

        def rebuild_history():
            chat_container.set_widgets([])
            for turn in conv.turns:
                add_message(turn["role"], turn["text"])

        def handle_command(text: str) -> bool:
            nonlocal last_user
            parts = text.split()
            cmd = parts[0].lower()

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

            if cmd == "/help":
                add_message(
                    "bot",
                    "Commands:\n"
                    "  /reset  — clear conversation\n"
                    "  /save   — save history to disk\n"
                    "  /retry  — regenerate last response\n"
                    "  /exit   — quit\n"
                    "  /help   — show this help",
                )
                return True

            return False

        def _do_generate(text: str, retry: bool = False):
            nonlocal last_user
            kb_ctx = kb.retrieve(text) if kb.loaded else ""

            if not retry:
                conv.add_user(text)
                last_user = text

            prompt = build_prompt(conv.recent(), text, fmt, kb_ctx)

            t0 = time.time()
            try:
                response = generate(model, tokenizer, prompt, args, device, fmt)
            except KeyboardInterrupt:
                status_label.value = "[dim]Interrupted.[/]"
                return

            ms = int((time.time() - t0) * 1000)
            conv.add_bot(response)
            add_message("bot", response)
            status_label.value = (
                f"[dim]{len(tokenizer.encode(response))} tok · {ms}ms[/]"
            )

        # ── Input handler ─────────────────────────────────────────────────────
        def on_submit():
            text = input_field.value.strip()
            if not text:
                return
            input_field.value = ""

            if text.startswith("/"):
                if handle_command(text):
                    return

            add_message("user", text)
            _do_generate(text)

        # Bind Enter on the InputField itself (has priority when focused)
        input_field.bind(ptg.keys.ENTER, lambda *_: on_submit())

        # ── Window layout ────────────────────────────────────────────────────
        window = ptg.Window(
            f"[bold 141]🤖 AI Chat[/]  [dim]{os.path.basename(args.model)}[/]",
            "",
            chat_container,
            "",
            status_label,
            "",
            input_field,
            overflow=ptg.Overflow.SCROLL,
            vertical_align=ptg.VerticalAlignment.BOTTOM,
            width=manager.terminal.width,
            height=manager.terminal.height,
            box="ROUNDED",
        )

        window.center()

        manager.layout.add_slot("Body")
        manager.add(window)

        # Focus the input field so Enter works immediately
        window.select(-1)


if __name__ == "__main__":
    main()
