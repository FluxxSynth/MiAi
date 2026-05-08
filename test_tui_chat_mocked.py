"""test_tui_chat_mocked.py — Test TUI chat with mocked model loading"""
import sys, os, unittest
from unittest.mock import MagicMock, patch, call
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestTUIChatMocked(unittest.TestCase):
    """Test tui_chat.py by mocking all heavy dependencies."""

    @classmethod
    def setUpClass(cls):
        # Mock torch and transformers before any import
        cls.torch_patcher = patch.dict(sys.modules, {
            "torch": MagicMock(),
            "torch.cuda": MagicMock(),
            "torch.cuda.amp": MagicMock(),
        })
        cls.torch_patcher.start()

        # Create mock torch module with is_available
        mock_torch = sys.modules["torch"]
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_torch.long = "long"
        mock_torch.no_grad.return_value.__enter__ = lambda *a: None
        mock_torch.no_grad.return_value.__exit__ = lambda *a: None
        mock_torch.full = MagicMock(return_value=MagicMock())
        mock_torch.zeros = MagicMock(return_value=MagicMock())

        # Mock transformers
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM = MagicMock()
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.BitsAndBytesConfig = MagicMock()
        sys.modules["transformers"] = mock_transformers

        # Mock peft
        mock_peft = MagicMock()
        mock_peft.PeftModel = MagicMock()
        sys.modules["peft"] = mock_peft
        sys.modules["bitsandbytes"] = MagicMock()

        # Mock pytermgui
        cls._setup_mock_pytermgui()

    @classmethod
    def _setup_mock_pytermgui(cls):
        """Build a minimal mock pytermgui with enough API surface."""
        mock_ptg = MagicMock()

        # Keys
        mock_keys = MagicMock()
        mock_keys.ENTER = "ENTER"
        mock_ptg.keys = mock_keys

        # Overflow / alignment
        mock_ptg.Overflow = MagicMock()
        mock_ptg.Overflow.SCROLL = "SCROLL"
        mock_ptg.VerticalAlignment = MagicMock()
        mock_ptg.VerticalAlignment.BOTTOM = "BOTTOM"

        # WindowManager as context manager
        mock_manager = MagicMock()
        mock_manager.terminal.width = 80
        mock_manager.terminal.height = 24
        mock_manager.layout.add_slot = MagicMock()
        mock_manager.add = MagicMock()
        mock_manager.stop = MagicMock()

        mock_wm_cls = MagicMock()
        mock_wm_cls.return_value.__enter__ = lambda *a: mock_manager
        mock_wm_cls.return_value.__exit__ = lambda *a: None
        mock_ptg.WindowManager = mock_wm_cls

        # Widgets
        mock_ptg.Window = MagicMock()
        mock_ptg.Container = MagicMock()
        mock_ptg.Label = MagicMock()
        mock_ptg.InputField = MagicMock()

        sys.modules["pytermgui"] = mock_ptg
        cls.mock_ptg = mock_ptg
        cls.mock_manager = mock_manager

    def _reload_chat(self):
        """Re-import chat.py with mocked deps."""
        modules_to_remove = [k for k in sys.modules if "chat" in k or "tui_chat" in k]
        for m in modules_to_remove:
            del sys.modules[m]

        import chat
        import tui_chat
        return chat, tui_chat

    @patch("builtins.print")
    @patch("os.path.exists")
    def test_tui_main_flow(self, mock_exists, mock_print):
        """Verify tui_chat.main() builds the UI and binds ENTER."""
        mock_exists.return_value = True

        chat, tui_chat = self._reload_chat()

        # Mock tokenizer + model
        mock_tok = MagicMock()
        mock_tok.pad_token = "[PAD]"
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 1
        mock_tok.encode = MagicMock(return_value=[1, 2, 3])
        mock_tok.decode = MagicMock(return_value="Hello there!")

        mock_model = MagicMock()
        # make generate() return a tensor-like object
        mock_out = MagicMock()
        mock_out.__getitem__ = MagicMock(return_value=mock_out)
        mock_out.shape = MagicMock()
        mock_out.shape.__getitem__ = MagicMock(return_value=5)
        mock_model.generate.return_value = [mock_out]

        with patch.object(chat, "load_model", return_value=(mock_model, mock_tok)):
            with patch.object(chat, "KnowledgeBase") as MockKB:
                MockKB.return_value.loaded = False
                with patch.object(chat, "Conversation") as MockConv:
                    mock_conv = MagicMock()
                    mock_conv.turns = []
                    mock_conv.recent.return_value = []
                    MockConv.return_value = mock_conv

                    # Mock parse_args
                    args = MagicMock()
                    args.model = "finetuned-model"
                    args.kb = None
                    args.history_file = "/tmp/test_hist.json"
                    args.no_qlora = True
                    args.temperature = 0.7
                    args.top_k = 50
                    args.top_p = 0.92
                    args.max_tokens = 512
                    args.beams = 1
                    args.rep_penalty = 1.15

                    with patch.object(chat, "parse_args", return_value=args):
                        tui_chat.main()

        # Assertions on PyTermGUI calls
        self.mock_ptg.Window.assert_called_once()
        window_call = self.mock_ptg.Window.call_args
        kwargs = window_call.kwargs if hasattr(window_call, "kwargs") else window_call[1]

        self.assertEqual(kwargs.get("overflow"), "SCROLL")
        self.assertEqual(kwargs.get("vertical_align"), "BOTTOM")

        # Verify WindowManager.add was called
        self.mock_manager.add.assert_called_once()

    @patch("builtins.print")
    @patch("os.path.exists")
    def test_model_not_found(self, mock_exists, mock_print):
        """Verify tui_chat exits when model path doesn't exist."""
        mock_exists.return_value = False

        chat, tui_chat = self._reload_chat()

        args = MagicMock()
        args.model = "nonexistent-model"

        with patch.object(chat, "parse_args", return_value=args):
            with self.assertRaises(SystemExit) as cm:
                tui_chat.main()
            self.assertEqual(cm.exception.code, 1)

    def test_prompt_formats(self):
        """Verify prompt formatting for all model families."""
        # Minimal chat import — we already have mocked deps
        import chat

        # Mistral
        p = chat.build_prompt([], "Hi", "mistral")
        self.assertIn("[INST]", p)
        self.assertIn("Hi", p)

        # LLaMA-3
        p = chat.build_prompt([], "Hi", "llama3")
        self.assertIn("<|begin_of_text|>", p)
        self.assertIn("Hi", p)

        # Phi-3
        p = chat.build_prompt([], "Hi", "phi3")
        self.assertIn("<|user|>", p)

        # Gemma
        p = chat.build_prompt([], "Hi", "gemma")
        self.assertIn("<start_of_turn>user", p)

    def test_conversation_roundtrip(self):
        """Verify Conversation add/pop/save roundtrip."""
        import chat
        import json

        test_path = "/tmp/tui_test_conv.json"
        if os.path.exists(test_path):
            os.remove(test_path)

        conv = chat.Conversation(test_path)
        conv.reset()
        conv.add_user("Test question")
        conv.add_bot("Test answer")
        conv.save()

        # Verify file contents
        with open(test_path) as f:
            data = json.load(f)
        self.assertEqual(len(data["turns"]), 2)
        self.assertEqual(data["turns"][0]["role"], "user")
        self.assertEqual(data["turns"][1]["role"], "bot")

        # Load fresh conversation
        conv2 = chat.Conversation(test_path)
        self.assertEqual(len(conv2.turns), 2)

    def test_detect_format(self):
        """Verify format detection logic."""
        import chat
        self.assertEqual(chat.detect_format("mistralai/Mistral-7B"), "mistral")
        self.assertEqual(chat.detect_format("meta-llama/Llama-3-8B"), "llama3")
        self.assertEqual(chat.detect_format("microsoft/Phi-3"), "phi3")
        self.assertEqual(chat.detect_format("google/gemma-2b"), "gemma")

    def test_stop_strings(self):
        """Verify stop string retrieval."""
        import chat
        self.assertIn("</s>", chat.get_stop_strings("mistral"))
        self.assertIn("<|eot_id|>", chat.get_stop_strings("llama3"))
        self.assertIn("<|end|>", chat.get_stop_strings("phi3"))
        self.assertIn("<end_of_turn>", chat.get_stop_strings("gemma"))

    def test_adapter_config_readable(self):
        """Verify the adapter config has required fields."""
        import json
        path = os.path.join(os.path.dirname(__file__), "finetuned-model", "adapter_config.json")
        self.assertTrue(os.path.exists(path), "adapter_config.json should exist")
        with open(path) as f:
            cfg = json.load(f)
        self.assertIn("base_model_name_or_path", cfg)
        self.assertTrue(len(cfg["base_model_name_or_path"]) > 0)


class TestSyntaxAndImports(unittest.TestCase):
    def test_tui_chat_compiles(self):
        """Verify tui_chat.py has no syntax errors."""
        import py_compile
        py_compile.compile(
            os.path.join(os.path.dirname(__file__), "tui_chat.py"),
            doraise=True,
        )

    def test_chat_compiles(self):
        """Verify chat.py has no syntax errors."""
        import py_compile
        py_compile.compile(
            os.path.join(os.path.dirname(__file__), "chat.py"),
            doraise=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
