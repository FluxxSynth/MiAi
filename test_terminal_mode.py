"""
test_terminal_mode.py — Tests for chat.py terminal mode and commands
====================================================================
Uses mocked model/tokenizer to test the terminal_loop function's command
handling and the parse_args function.
"""
import os, sys, unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Module-level mocks
# ---------------------------------------------------------------------------
def _setup_mocks():
    for k in list(sys.modules.keys()):
        if "chat" in k:
            del sys.modules[k]

    patchers = [
        patch.dict(sys.modules, {
            "torch": MagicMock(),
            "torch.cuda": MagicMock(),
            "torch.cuda.amp": MagicMock(),
        }),
        patch.dict(sys.modules, {
            "transformers": MagicMock(),
        }),
        patch.dict(sys.modules, {"peft": MagicMock()}),
        patch.dict(sys.modules, {"bitsandbytes": MagicMock()}),
    ]
    for p in patchers:
        p.start()
    return patchers


_patchers = []


def setUpModule():
    global _patchers
    _patchers = _setup_mocks()
    import chat
    # Force parse_args to use known test args
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float16 = "float16"
    mock_torch.long = "long"
    mock_torch.no_grad.return_value.__enter__ = lambda *a: None
    mock_torch.no_grad.return_value.__exit__ = lambda *a: None
    mock_torch.full = MagicMock(return_value=MagicMock())


def tearDownModule():
    for p in _patchers:
        p.stop()


class TestParseArgs(unittest.TestCase):
    """Test CLI argument parsing."""

    def test_default_values(self):
        """parse_args() should return expected defaults."""
        import chat
        with patch.object(sys, "argv", ["chat.py"]):
            args = chat.parse_args()
        self.assertEqual(args.model, "finetuned-model")
        self.assertEqual(args.temperature, 0.7)
        self.assertEqual(args.top_k, 50)
        self.assertEqual(args.top_p, 0.92)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.beams, 1)
        self.assertEqual(args.rep_penalty, 1.15)
        self.assertIsNone(args.kb)
        self.assertEqual(args.history_file, "chat_history.json")
        self.assertEqual(args.port, 5000)
        self.assertFalse(args.no_web)
        self.assertFalse(args.no_qlora)

    def test_custom_model_flag(self):
        """--model flag should override default."""
        import chat
        with patch.object(sys, "argv", ["chat.py", "--model", "my-custom-model"]):
            args = chat.parse_args()
        self.assertEqual(args.model, "my-custom-model")

    def test_no_web_flag(self):
        """--no_web flag should be True when passed."""
        import chat
        with patch.object(sys, "argv", ["chat.py", "--no_web"]):
            args = chat.parse_args()
        self.assertTrue(args.no_web)

    def test_temperature_flag(self):
        """--temperature should accept float values."""
        import chat
        with patch.object(sys, "argv", ["chat.py", "--temperature", "0.3"]):
            args = chat.parse_args()
        self.assertAlmostEqual(args.temperature, 0.3)

    def test_kb_flag(self):
        """--kb should set knowledge base folder."""
        import chat
        with patch.object(sys, "argv", ["chat.py", "--kb", "./docs"]):
            args = chat.parse_args()
        self.assertEqual(args.kb, "./docs")

    def test_beams_flag(self):
        """--beams should accept integer."""
        import chat
        with patch.object(sys, "argv", ["chat.py", "--beams", "3"]):
            args = chat.parse_args()
        self.assertEqual(args.beams, 3)

    def test_port_flag(self):
        """--port should set the web server port."""
        import chat
        with patch.object(sys, "argv", ["chat.py", "--port", "8080"]):
            args = chat.parse_args()
        self.assertEqual(args.port, 8080)


class TestTerminalLoop(unittest.TestCase):
    """Test terminal_loop function command handling."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.chat = chat

    def setUp(self):
        chat = self.chat
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token = "[PAD]"
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "Test response"

        self.mock_model = MagicMock()
        mock_out = MagicMock()
        mock_out.__getitem__ = MagicMock(return_value=mock_out)
        mock_out.shape.__getitem__ = MagicMock(return_value=3)
        self.mock_model.generate.return_value = [mock_out]

        self.mock_conv = MagicMock()
        self.mock_conv.turns = []
        self.mock_conv.persona = "assistant"
        self.mock_conv.recent.return_value = []

        self.mock_kb = MagicMock()
        self.mock_kb.loaded = False

        self.args = MagicMock(
            model="finetuned-model",
            temperature=0.7,
            top_k=50,
            top_p=0.92,
            max_tokens=512,
            beams=1,
            rep_penalty=1.15,
            kb=None,
            history_file="/tmp/test_hist.json",
            port=5000,
            no_web=False,
            no_qlora=True,
        )

    @patch("builtins.input")
    @patch("builtins.print")
    def test_exit_command(self, mock_print, mock_input):
        """Typing 'exit' should exit the loop."""
        mock_input.side_effect = ["exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )

    @patch("builtins.input")
    @patch("builtins.print")
    def test_quit_command(self, mock_print, mock_input):
        """Typing 'quit' should exit the loop."""
        mock_input.side_effect = ["quit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )

    @patch("builtins.input")
    @patch("builtins.print")
    def test_empty_input_ignored(self, mock_print, mock_input):
        """Empty input should be skipped without error."""
        mock_input.side_effect = ["", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )

    @patch("builtins.input")
    @patch("builtins.print")
    def test_whitespace_input_ignored(self, mock_print, mock_input):
        """Whitespace-only input should be skipped."""
        mock_input.side_effect = ["   ", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )

    @patch("builtins.input")
    @patch("builtins.print")
    def test_reset_command(self, mock_print, mock_input):
        """/reset should call conv.reset()."""
        mock_input.side_effect = ["/reset", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )
        self.mock_conv.reset.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_save_command(self, mock_print, mock_input):
        """/save should call conv.save()."""
        mock_input.side_effect = ["/save", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )
        self.mock_conv.save.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_retry_without_last_user(self, mock_print, mock_input):
        """/retry with no last_user should not crash."""
        # Set last_user to empty by not sending a user message first
        mock_input.side_effect = ["/retry", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )

    @patch("builtins.input")
    @patch("builtins.print")
    def test_beams_command(self, mock_print, mock_input):
        """/beams N should update args.beams."""
        mock_input.side_effect = ["/beams 3", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )
        self.assertEqual(self.args.beams, 3)

    @patch("builtins.input")
    @patch("builtins.print")
    def test_unknown_command_handled(self, mock_print, mock_input):
        """Unknown command should not crash (just pass through as message)."""
        mock_input.side_effect = ["/some_unknown_cmd", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )

    @patch("builtins.input")
    @patch("builtins.print")
    def test_normal_message_flow(self, mock_print, mock_input):
        """Normal message should trigger generate and add bot response."""
        mock_input.side_effect = ["Hello bot", "exit"]
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )
        self.mock_conv.add_user.assert_called_once_with("Hello bot")
        self.mock_conv.add_bot.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_keyboard_interrupt(self, mock_print, mock_input):
        """Ctrl+C (KeyboardInterrupt) should exit gracefully."""
        mock_input.side_effect = KeyboardInterrupt()
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )
        # Should not raise

    @patch("builtins.input")
    @patch("builtins.print")
    def test_eoferror_handling(self, mock_print, mock_input):
        """EOFError (Ctrl+D) should exit gracefully."""
        mock_input.side_effect = EOFError()
        self.chat.terminal_loop(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
