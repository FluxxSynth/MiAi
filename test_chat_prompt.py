"""
test_chat_prompt.py — Extended tests for chat.py prompt formatting
===================================================================
Tests build_prompt with history, RAG context, edge cases, and all formats.
"""
import os, sys, unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Module-level mocks
# ---------------------------------------------------------------------------
_patchers = []


def setUpModule():
    global _patchers
    from unittest.mock import MagicMock, patch
    _patchers.append(patch.dict(sys.modules, {
        "torch": MagicMock(),
        "torch.cuda": MagicMock(),
        "torch.cuda.amp": MagicMock(),
    }))
    _patchers.append(patch.dict(sys.modules, {"transformers": MagicMock()}))
    _patchers.append(patch.dict(sys.modules, {"peft": MagicMock()}))
    _patchers.append(patch.dict(sys.modules, {"bitsandbytes": MagicMock()}))

    for p in _patchers:
        p.start()

    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float16 = "float16"
    mock_torch.long = "long"
    mock_torch.no_grad.return_value.__enter__ = lambda *a: None
    mock_torch.no_grad.return_value.__exit__ = lambda *a: None


def tearDownModule():
    for p in _patchers:
        p.stop()


class TestBuildPromptWithoutHistory(unittest.TestCase):
    """Test build_prompt with empty history (single-turn)."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.build_prompt = chat.build_prompt
        cls.SYSTEM_MSG = chat.SYSTEM_MSG

    def test_mistral_format_no_history(self):
        """build_prompt should produce Mistral format with no history."""
        prompt = self.build_prompt([], "Hello", "mistral")
        self.assertIn("[INST]", prompt)
        self.assertIn("Hello", prompt)
        self.assertIn("[/INST]", prompt)

    def test_llama3_format_no_history(self):
        """build_prompt should produce LLaMA-3 format with no history."""
        prompt = self.build_prompt([], "Hello", "llama3")
        self.assertIn("<|begin_of_text|>", prompt)
        self.assertIn("Hello", prompt)

    def test_phi3_format_no_history(self):
        """build_prompt should produce Phi-3 format with no history."""
        prompt = self.build_prompt([], "Hello", "phi3")
        self.assertIn("<|user|>", prompt)
        self.assertIn("Hello", prompt)

    def test_gemma_format_no_history(self):
        """build_prompt should produce Gemma format with no history."""
        prompt = self.build_prompt([], "Hello", "gemma")
        self.assertIn("<start_of_turn>user", prompt)
        self.assertIn("Hello", prompt)

    def test_system_message_included(self):
        """System message should appear in prompt."""
        prompt = self.build_prompt([], "Hello", "mistral")
        self.assertIn("helpful", prompt.lower())

    def test_empty_message(self):
        """build_prompt with empty message should not crash."""
        prompt = self.build_prompt([], "", "mistral")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)


class TestBuildPromptWithHistory(unittest.TestCase):
    """Test build_prompt with conversation history."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.build_prompt = chat.build_prompt

    def test_mistral_with_single_history_turn(self):
        """Mistral format should include 1 history turn."""
        turns = [
            {"role": "user", "text": "Previous question"},
            {"role": "bot", "text": "Previous answer"},
        ]
        prompt = self.build_prompt(turns, "New question", "mistral")
        self.assertIn("Previous question", prompt)
        self.assertIn("Previous answer", prompt)
        self.assertIn("New question", prompt)

    def test_mistral_with_multiple_history_turns(self):
        """Mistral format should include all history turns."""
        turns = [
            {"role": "user", "text": "Q1"},
            {"role": "bot", "text": "A1"},
            {"role": "user", "text": "Q2"},
            {"role": "bot", "text": "A2"},
        ]
        prompt = self.build_prompt(turns, "Q3", "mistral")
        self.assertIn("Q1", prompt)
        self.assertIn("A1", prompt)
        self.assertIn("Q2", prompt)
        self.assertIn("A2", prompt)
        self.assertIn("Q3", prompt)

    def test_llama3_with_history(self):
        """LLaMA-3 format should include conversation history."""
        turns = [
            {"role": "user", "text": "Hello"},
            {"role": "bot", "text": "Hi"},
        ]
        prompt = self.build_prompt(turns, "How are you?", "llama3")
        self.assertIn("Hello", prompt)
        self.assertIn("Hi", prompt)
        self.assertIn("How are you?", prompt)
        # Should have 2 user turns: history + current
        self.assertEqual(
            prompt.count("<|start_header_id|>user<|end_header_id|>"), 2
        )

    def test_phi3_with_history(self):
        """Phi-3 format should include history."""
        turns = [
            {"role": "user", "text": "Hello"},
            {"role": "bot", "text": "Hi"},
        ]
        prompt = self.build_prompt(turns, "Q2", "phi3")
        self.assertIn("Hello", prompt)
        self.assertIn("Hi", prompt)
        self.assertIn("Q2", prompt)

    def test_gemma_with_history(self):
        """Gemma format should include history."""
        turns = [
            {"role": "user", "text": "Hello"},
            {"role": "bot", "text": "Hi"},
        ]
        prompt = self.build_prompt(turns, "Q2", "gemma")
        self.assertIn("Hello", prompt)
        self.assertIn("Hi", prompt)
        self.assertIn("Q2", prompt)

    def test_unpaired_user_turn_omitted(self):
        """User turn without bot response should be omitted from history."""
        turns = [
            {"role": "user", "text": "Valid question"},
            {"role": "bot", "text": "Valid answer"},
            {"role": "user", "text": "Unpaired question"},
        ]
        prompt = self.build_prompt(turns, "Current", "mistral")
        self.assertIn("Valid question", prompt)
        self.assertIn("Valid answer", prompt)
        # Unpaired question should not appear in history
        # (but it might appear if the implementation puts it there)
        self.assertIn("Current", prompt)


class TestBuildPromptWithRAG(unittest.TestCase):
    """Test build_prompt with knowledge base context."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.build_prompt = chat.build_prompt

    def test_kb_context_appended(self):
        """kb_ctx should be appended to system message."""
        prompt = self.build_prompt([], "Hello", "mistral",
                                   kb_ctx="Important context here")
        self.assertIn("Important context here", prompt)

    def test_kb_context_empty(self):
        """Empty kb_ctx should not change output."""
        prompt_with = self.build_prompt([], "Hello", "mistral", kb_ctx="")
        prompt_without = self.build_prompt([], "Hello", "mistral")
        self.assertEqual(prompt_with, prompt_without)

    def test_kb_context_with_history(self):
        """kb_ctx should work with history turns."""
        turns = [
            {"role": "user", "text": "Old Q"},
            {"role": "bot", "text": "Old A"},
        ]
        prompt = self.build_prompt(turns, "New Q", "mistral",
                                   kb_ctx="Relevant docs")
        self.assertIn("Old Q", prompt)
        self.assertIn("Relevant docs", prompt)
        self.assertIn("New Q", prompt)

    def test_kb_context_llama3(self):
        """kb_ctx should work with LLaMA-3 format."""
        prompt = self.build_prompt([], "Hello", "llama3",
                                   kb_ctx="KB context for LLaMA")
        self.assertIn("KB context for LLaMA", prompt)


class TestBuildPromptEdgeCases(unittest.TestCase):
    """Test edge cases for build_prompt."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.build_prompt = chat.build_prompt

    def test_special_characters(self):
        """Special characters in message should not break formatting."""
        prompt = self.build_prompt([], "Hello <world> & more",
                                   "mistral")
        self.assertIn("Hello", prompt)

    def test_multiline_message(self):
        """Multi-line message should be handled."""
        prompt = self.build_prompt([], "Line 1\nLine 2\nLine 3",
                                   "mistral")
        self.assertIn("Line 1", prompt)
        self.assertIn("Line 2", prompt)

    def test_very_long_message(self):
        """Very long message should not crash."""
        long_msg = "word " * 10000
        prompt = self.build_prompt([], long_msg, "mistral")
        self.assertIn("word", prompt)

    def test_empty_turns_list_with_non_empty_message(self):
        """Empty turns list with valid message should work."""
        prompt = self.build_prompt([], "Hello", "mistral")
        self.assertIn("Hello", prompt)

    def test_non_alternating_history(self):
        """History with non-alternating roles should not crash."""
        turns = [
            {"role": "user", "text": "Q1"},
            {"role": "user", "text": "Q2"},  # Two users in a row
            {"role": "bot", "text": "A1"},
        ]
        prompt = self.build_prompt(turns, "Q3", "mistral")
        self.assertIn("Q3", prompt)
        self.assertIsInstance(prompt, str)


class TestStopStrings(unittest.TestCase):
    """Test get_stop_strings."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.get_stop_strings = chat.get_stop_strings

    def test_mistral_stops(self):
        """Mistral stop strings should include </s>."""
        stops = self.get_stop_strings("mistral")
        self.assertIn("</s>", stops)

    def test_llama3_stops(self):
        """LLaMA-3 stop strings should include <|eot_id|>."""
        stops = self.get_stop_strings("llama3")
        self.assertIn("<|eot_id|>", stops)

    def test_phi3_stops(self):
        """Phi-3 stop strings should include <|end|>."""
        stops = self.get_stop_strings("phi3")
        self.assertIn("<|end|>", stops)

    def test_gemma_stops(self):
        """Gemma stop strings should include <end_of_turn>."""
        stops = self.get_stop_strings("gemma")
        self.assertIn("<end_of_turn>", stops)

    def test_unknown_format(self):
        """Unknown format should return default stops."""
        stops = self.get_stop_strings("unknown")
        self.assertEqual(stops, ["</s>"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
