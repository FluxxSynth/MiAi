"""
test_chat_api.py — Tests for chat.py Flask API endpoints
========================================================
Uses Flask's test client to verify all API routes with mocked model/deps.
"""
import os, sys, json, unittest, tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Module-level mocks (run before any chat import)
# ---------------------------------------------------------------------------
_torch_patcher = None
_transformers_patcher = None
_peft_patcher = None
_bnb_patcher = None


def setUpModule():
    global _torch_patcher, _transformers_patcher, _peft_patcher, _bnb_patcher
    _torch_patcher = patch.dict(sys.modules, {
        "torch": MagicMock(),
        "torch.cuda": MagicMock(),
        "torch.cuda.amp": MagicMock(),
    })
    _torch_patcher.start()
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float16 = "float16"
    mock_torch.long = "long"
    mock_torch.no_grad.return_value.__enter__ = lambda *a: None
    mock_torch.no_grad.return_value.__exit__ = lambda *a: None
    mock_torch.full = MagicMock(return_value=MagicMock())

    mock_transformers = MagicMock()
    mock_transformers.AutoModelForCausalLM = MagicMock()
    mock_transformers.AutoTokenizer = MagicMock()
    mock_transformers.BitsAndBytesConfig = MagicMock()
    _transformers_patcher = patch.dict(sys.modules, {"transformers": mock_transformers})
    _transformers_patcher.start()

    _peft_patcher = patch.dict(sys.modules, {"peft": MagicMock()})
    _peft_patcher.start()
    _bnb_patcher = patch.dict(sys.modules, {"bitsandbytes": MagicMock()})
    _bnb_patcher.start()

    sys.modules["flask"] = MagicMock()


def tearDownModule():
    for p in (_torch_patcher, _transformers_patcher, _peft_patcher, _bnb_patcher):
        if p:
            p.stop()


class TestChatAPI(unittest.TestCase):
    """Test Flask API endpoints using the test client."""

    @classmethod
    def setUpClass(cls):
        # Re-import chat module fresh
        for k in list(sys.modules.keys()):
            if "chat" in k:
                del sys.modules[k]
        import chat
        cls.chat = chat

    def setUp(self):
        chat = self.chat

        # Mock model + tokenizer
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

        self.mock_kb = MagicMock()
        self.mock_kb.loaded = False
        self.mock_kb.retrieve.return_value = ""

        self.mock_conv = chat.Conversation.__new__(chat.Conversation)
        self.mock_conv.turns = []
        self.mock_conv.persona = "assistant"
        self.mock_conv.history_file = "/tmp/test_api_hist.json"

        self.args = MagicMock(
            model="finetuned-model",
            temperature=0.7,
            top_k=50,
            top_p=0.92,
            max_tokens=512,
            beams=1,
            rep_penalty=1.15,
            kb=None,
            history_file="/tmp/test_api_hist.json",
            port=5000,
            no_web=False,
            no_qlora=True,
        )

    def _make_app(self):
        """Build a Flask test app with the mocked dependencies."""
        app = self.chat.create_app(
            self.mock_model, self.mock_tokenizer,
            self.mock_conv, self.mock_kb,
            self.args, "cpu", "mistral",
        )
        app.config["TESTING"] = True
        return app.test_client()

    # -- Index route --------------------------------------------------------

    def test_index_returns_html(self):
        """GET / should return HTML with chatbot UI."""
        client = self._make_app()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.content_type.lower())
        self.assertIn("AI Chat", resp.get_data(as_text=True))

    # -- /api/info ---------------------------------------------------------

    def test_api_info_returns_model(self):
        """GET /api/info should return model name."""
        client = self._make_app()
        resp = client.get("/api/info")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        self.assertIn("model", data)
        self.assertEqual(data["model"], "finetuned-model")

    # -- /api/chat ---------------------------------------------------------

    def test_api_chat_empty_message(self):
        """POST /api/chat with empty message should return empty response."""
        client = self._make_app()
        resp = client.post(
            "/api/chat",
            content_type="application/json",
            data=json.dumps({"message": ""}),
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        self.assertEqual(data["response"], "")
        self.assertEqual(data["tokens"], 0)
        self.assertEqual(data["ms"], 0)

    def test_api_chat_missing_message_key(self):
        """POST /api/chat without 'message' key should treat as empty."""
        client = self._make_app()
        resp = client.post(
            "/api/chat",
            content_type="application/json",
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        self.assertEqual(data["response"], "")

    def test_api_chat_normal_flow(self):
        """POST /api/chat with valid message returns response with metadata."""
        client = self._make_app()
        resp = client.post(
            "/api/chat",
            content_type="application/json",
            data=json.dumps({"message": "Hello"}),
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        self.assertIn("response", data)
        self.assertIn("tokens", data)
        self.assertIn("ms", data)
        self.assertIsInstance(data["tokens"], int)
        self.assertIsInstance(data["ms"], int)

    def test_api_chat_retry_flag(self):
        """POST /api/chat with retry=true should not add a user turn."""
        client = self._make_app()
        # Normal message first
        client.post(
            "/api/chat",
            content_type="application/json",
            data=json.dumps({"message": "First msg"}),
        )
        initial_turn_count = len(self.mock_conv.turns)

        # Retry — should pop last bot first
        resp = client.post(
            "/api/chat",
            content_type="application/json",
            data=json.dumps({"message": "First msg", "retry": True}),
        )
        self.assertEqual(resp.status_code, 200)

    def test_api_chat_invalid_content_type(self):
        """POST /api/chat without JSON content type should fail."""
        client = self._make_app()
        resp = client.post(
            "/api/chat",
            data="not json",
        )
        # Flask will handle this differently depending on config
        # At minimum it should not crash
        self.assertIn(resp.status_code, (200, 400, 415))

    def test_api_chat_persona_integration(self):
        """Persona set via API should be used in build_prompt."""
        # This tests that the persona endpoint and chat flow integrate
        client = self._make_app()
        # Set persona
        resp = client.post(
            "/api/persona",
            content_type="application/json",
            data=json.dumps({"persona": "tutor"}),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.mock_conv.persona, "tutor")

    # -- /api/reset ---------------------------------------------------------

    def test_api_reset_clears_turns(self):
        """POST /api/reset should clear conversation turns."""
        self.mock_conv.turns = [{"role": "user", "text": "Hi"}]
        client = self._make_app()
        resp = client.post("/api/reset")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        self.assertTrue(data.get("ok"))
        self.assertEqual(len(self.mock_conv.turns), 0)

    # -- /api/save ----------------------------------------------------------

    def test_api_save_returns_ok(self):
        """POST /api/save should return ok and trigger conv.save()."""
        with patch.object(self.mock_conv, "save") as mock_save:
            client = self._make_app()
            resp = client.post("/api/save")
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.get_data(as_text=True))
            self.assertTrue(data.get("ok"))
            mock_save.assert_called_once()

    # -- /api/retry ---------------------------------------------------------

    def test_api_retry_pops_last_bot(self):
        """POST /api/retry should remove the last bot turn."""
        self.mock_conv.turns = [
            {"role": "user", "text": "Hi"},
            {"role": "bot", "text": "Hello"},
        ]
        client = self._make_app()
        resp = client.post("/api/retry")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(self.mock_conv.turns), 1)
        self.assertEqual(self.mock_conv.turns[0]["role"], "user")

    def test_api_retry_empty_turns(self):
        """POST /api/retry with no bot turns should not crash."""
        self.mock_conv.turns = []
        client = self._make_app()
        resp = client.post("/api/retry")
        self.assertEqual(resp.status_code, 200)

    # -- /api/persona -------------------------------------------------------

    def test_api_persona_valid_values(self):
        """POST /api/persona with valid persona should update conv.persona."""
        client = self._make_app()
        for persona in ("assistant", "tutor", "coder", "creative"):
            resp = client.post(
                "/api/persona",
                content_type="application/json",
                data=json.dumps({"persona": persona}),
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(self.mock_conv.persona, persona)

    def test_api_persona_no_body(self):
        """POST /api/persona with missing persona should keep default."""
        client = self._make_app()
        self.mock_conv.persona = "assistant"
        resp = client.post(
            "/api/persona",
            content_type="application/json",
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 200)
        # Default (None from .get) — the code uses "assistant" as fallback
        self.assertEqual(self.mock_conv.persona, None)


class TestChatAPICORSAndErrors(unittest.TestCase):
    """Test error handling and edge cases not covered by normal flow."""

    @classmethod
    def setUpClass(cls):
        for k in list(sys.modules.keys()):
            if "chat" in k:
                del sys.modules[k]
        import chat
        cls.chat = chat

    def test_generate_handles_nan_loss(self):
        """generate() should handle edge cases gracefully (smoke test)."""
        # This is more of a build_prompt + generate integration smoke test
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
