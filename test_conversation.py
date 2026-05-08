"""
test_conversation.py — Extended tests for the Conversation class
=================================================================
Tests beyond the basic roundtrip covered in test_tui_chat_mocked.py.
"""
import os, sys, json, unittest, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Module-level mocks
# ---------------------------------------------------------------------------
_torch_patcher = None
_transformers_patcher = None
_peft_patcher = None
_bnb_patcher = None


def setUpModule():
    global _torch_patcher, _transformers_patcher, _peft_patcher, _bnb_patcher
    from unittest.mock import MagicMock, patch

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


def tearDownModule():
    for p in (_torch_patcher, _transformers_patcher, _peft_patcher, _bnb_patcher):
        if p:
            p.stop()


class TestConversation(unittest.TestCase):
    """Test Conversation class methods."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.Conversation = chat.Conversation

    def setUp(self):
        self.history_file = tempfile.mktemp(suffix=".json")

    def tearDown(self):
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

    # -- Construction / Init -----------------------------------------------

    def test_new_conversation_empty(self):
        """New Conversation should have no turns and default persona."""
        conv = self.Conversation.__new__(self.Conversation)
        conv.history_file = self.history_file
        conv.turns = []
        conv.persona = "assistant"
        conv._load()
        self.assertEqual(conv.turns, [])
        self.assertEqual(conv.persona, "assistant")

    def test_init_loads_existing_file(self):
        """Conversation should load existing history file."""
        data = {"turns": [
            {"role": "user", "text": "Hello", "ts": "12:00"},
            {"role": "bot", "text": "Hi", "ts": "12:00"},
        ]}
        with open(self.history_file, "w") as f:
            json.dump(data, f)
        conv = self.Conversation(self.history_file)
        self.assertEqual(len(conv.turns), 2)
        self.assertEqual(conv.turns[0]["role"], "user")
        self.assertEqual(conv.turns[1]["role"], "bot")

    def test_init_loads_corrupt_json_gracefully(self):
        """Conversation should handle corrupt JSON without crashing."""
        with open(self.history_file, "w") as f:
            f.write("not valid json{{{")
        conv = self.Conversation(self.history_file)
        self.assertEqual(conv.turns, [])

    def test_init_loads_nonexistent_file(self):
        """Conversation should handle missing file gracefully."""
        conv = self.Conversation("/tmp/does_not_exist_xyz.json")
        self.assertEqual(conv.turns, [])

    def test_init_loads_empty_json(self):
        """Conversation should handle empty JSON object."""
        with open(self.history_file, "w") as f:
            json.dump({}, f)
        conv = self.Conversation(self.history_file)
        self.assertEqual(conv.turns, [])

    def test_init_loads_missing_turns_key(self):
        """Conversation should handle JSON missing 'turns' key."""
        with open(self.history_file, "w") as f:
            json.dump({"other_key": []}, f)
        conv = self.Conversation(self.history_file)
        self.assertEqual(conv.turns, [])

    # -- add_user / add_bot ------------------------------------------------

    def test_add_user_appends_turn(self):
        """add_user should append a user turn."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Hello")
        self.assertEqual(len(conv.turns), 1)
        self.assertEqual(conv.turns[0]["role"], "user")
        self.assertEqual(conv.turns[0]["text"], "Hello")

    def test_add_bot_appends_turn(self):
        """add_bot should append a bot turn."""
        conv = self.Conversation(self.history_file)
        conv.add_bot("Hi there")
        self.assertEqual(len(conv.turns), 1)
        self.assertEqual(conv.turns[0]["role"], "bot")
        self.assertEqual(conv.turns[0]["text"], "Hi there")

    def test_turns_have_timestamp(self):
        """Added turns should have a timestamp."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Hello")
        self.assertIn("ts", conv.turns[0])
        self.assertIsInstance(conv.turns[0]["ts"], str)
        self.assertGreater(len(conv.turns[0]["ts"]), 0)

    # -- reset -------------------------------------------------------------

    def test_reset_clears_turns(self):
        """reset() should clear all turns."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Hello")
        conv.add_bot("Hi")
        conv.reset()
        self.assertEqual(conv.turns, [])

    def test_reset_empty_conv(self):
        """reset() on empty conversation should not crash."""
        conv = self.Conversation(self.history_file)
        conv.reset()
        self.assertEqual(conv.turns, [])

    # -- pop_last_bot ------------------------------------------------------

    def test_pop_last_bot_removes_bot(self):
        """pop_last_bot() should remove the last bot turn."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Hello")
        conv.add_bot("Hi")
        conv.add_user("Question 2")
        conv.add_bot("Answer 2")
        conv.pop_last_bot()
        self.assertEqual(len(conv.turns), 3)
        self.assertEqual(conv.turns[-1]["role"], "user")

    def test_pop_last_bot_no_bot(self):
        """pop_last_bot() with no bot turns should not crash."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Hello")
        conv.pop_last_bot()  # Should not crash
        self.assertEqual(len(conv.turns), 1)

    def test_pop_last_bot_empty_conv(self):
        """pop_last_bot() on empty conversation should not crash."""
        conv = self.Conversation(self.history_file)
        conv.pop_last_bot()

    def test_pop_last_bot_removes_last_bot_in_sequence(self):
        """pop_last_bot() should remove the LAST bot, not first."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Hello")
        conv.add_bot("Goodbye")
        conv.add_user("Another question")
        conv.add_bot("Another answer")
        conv.pop_last_bot()
        self.assertEqual(len(conv.turns), 3)
        self.assertEqual(conv.turns[-1]["text"], "Another question")

    # -- save / load roundtrip ---------------------------------------------

    def test_save_and_load_maintains_order(self):
        """Save then load should maintain turn order."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Q1")
        conv.add_bot("A1")
        conv.add_user("Q2")
        conv.add_bot("A2")
        conv.save()

        conv2 = self.Conversation(self.history_file)
        self.assertEqual(len(conv2.turns), 4)
        self.assertEqual(conv2.turns[0]["text"], "Q1")
        self.assertEqual(conv2.turns[1]["text"], "A1")
        self.assertEqual(conv2.turns[2]["text"], "Q2")
        self.assertEqual(conv2.turns[3]["text"], "A2")

    def test_save_with_empty_conv(self):
        """save() on empty conversation should create valid JSON."""
        conv = self.Conversation(self.history_file)
        conv.save()
        with open(self.history_file) as f:
            data = json.load(f)
        self.assertEqual(data["turns"], [])

    def test_load_after_reset(self):
        """After reset and save, loading should show empty turns."""
        conv = self.Conversation(self.history_file)
        conv.add_user("Test")
        conv.add_bot("Response")
        conv.save()
        conv.reset()
        conv.save()

        conv2 = self.Conversation(self.history_file)
        self.assertEqual(len(conv2.turns), 0)

    # -- recent() ----------------------------------------------------------

    def test_recent_with_fewer_turns_than_max(self):
        """recent() should return all turns when under max."""
        conv = self.Conversation(self.history_file)
        for i in range(3):
            conv.add_user(f"Q{i}")
            conv.add_bot(f"A{i}")
        recent = conv.recent(max_turns=10)
        self.assertEqual(len(recent), 6)

    def test_recent_with_more_turns_than_max(self):
        """recent() should cap at max_turns*2."""
        conv = self.Conversation(self.history_file)
        for i in range(20):
            conv.add_user(f"Q{i}")
            conv.add_bot(f"A{i}")
        recent = conv.recent(max_turns=5)
        self.assertLessEqual(len(recent), 10)

    def test_recent_with_zero_turns(self):
        """recent() on empty conversation should return empty list."""
        conv = self.Conversation(self.history_file)
        self.assertEqual(conv.recent(), [])

    def test_recent_returns_most_recent_turns(self):
        """recent() should return the most recent turns."""
        conv = self.Conversation(self.history_file)
        for i in range(10):
            conv.add_user(f"Q{i}")
            conv.add_bot(f"A{i}")
        recent = conv.recent(max_turns=3)
        self.assertEqual(len(recent), 6)
        self.assertIn("Q7", recent[0]["text"])
        self.assertIn("A9", recent[-1]["text"])

    # -- persona -----------------------------------------------------------

    def test_persona_default(self):
        """Default persona should be 'assistant'."""
        conv = self.Conversation(self.history_file)
        self.assertEqual(conv.persona, "assistant")

    def test_persona_can_be_changed(self):
        """Persona should be settable."""
        conv = self.Conversation(self.history_file)
        conv.persona = "tutor"
        self.assertEqual(conv.persona, "tutor")


if __name__ == "__main__":
    unittest.main(verbosity=2)
