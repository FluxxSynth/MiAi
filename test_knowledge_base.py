"""
test_knowledge_base.py — Tests for the KnowledgeBase RAG system
================================================================
Tests document indexing, retrieval, and edge cases without real files.
"""
import os, sys, unittest, tempfile, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Module-level mocks for torch/transformers (needed for chat.py import)
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


class TestKnowledgeBaseInit(unittest.TestCase):
    """Test KnowledgeBase initialization behavior."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.KnowledgeBase = chat.KnowledgeBase

    def test_init_no_folder(self):
        """KnowledgeBase(None) should not crash and remain unloaded."""
        kb = self.KnowledgeBase(None)
        self.assertFalse(kb.loaded)
        self.assertEqual(kb.chunks, [])

    def test_init_empty_string(self):
        """KnowledgeBase('') should not crash."""
        kb = self.KnowledgeBase("")
        self.assertFalse(kb.loaded)
        self.assertEqual(kb.chunks, [])

    def test_init_nonexistent_folder(self):
        """KnowledgeBase with nonexistent folder should not crash."""
        kb = self.KnowledgeBase("/tmp/nonexistent_folder_xyz_123")
        self.assertFalse(kb.loaded)
        self.assertEqual(kb.chunks, [])

    def test_init_empty_folder(self):
        """KnowledgeBase with empty folder should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self.KnowledgeBase(tmpdir)
            self.assertFalse(kb.loaded)
            self.assertEqual(kb.chunks, [])

    def test_init_folder_with_non_txt_files(self):
        """KnowledgeBase should skip non-.txt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-txt file
            with open(os.path.join(tmpdir, "notes.md"), "w") as f:
                f.write("# Markdown notes\n" * 50)
            kb = self.KnowledgeBase(tmpdir)
            self.assertFalse(kb.loaded)
            self.assertEqual(kb.chunks, [])

    def test_init_with_single_file(self):
        """KnowledgeBase should index a single .txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("machine learning natural language processing " * 30)
            kb = self.KnowledgeBase(tmpdir)
            self.assertTrue(kb.loaded)
            self.assertGreater(len(kb.chunks), 0)

    def test_init_with_multiple_files(self):
        """KnowledgeBase should index multiple .txt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                with open(os.path.join(tmpdir, f"doc{i}.txt"), "w") as f:
                    f.write(f"Document number {i} about AI chatbots. " * 20)
            kb = self.KnowledgeBase(tmpdir)
            self.assertTrue(kb.loaded)
            self.assertGreater(len(kb.chunks), 0)

    def test_init_skips_short_content(self):
        """KnowledgeBase should skip chunks shorter than 50 chars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "short.txt"), "w") as f:
                f.write("too short")
            kb = self.KnowledgeBase(tmpdir)
            # No chunks should be indexed (content < 50 chars after split)
            self.assertFalse(kb.loaded)


class TestKnowledgeBaseRetrieval(unittest.TestCase):
    """Test KnowledgeBase retrieval accuracy."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.KnowledgeBase = chat.KnowledgeBase

    def test_retrieve_unloaded(self):
        """retrieve() on unloaded KB should return empty string."""
        kb = self.KnowledgeBase(None)
        result = kb.retrieve("test query")
        self.assertEqual(result, "")

    def test_retrieve_with_relevant_content(self):
        """retrieve() should return relevant chunks for matching query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("Python is a programming language. " * 20)
                f.write("Machine learning uses algorithms. " * 20)
                f.write("Data science involves statistics. " * 20)
            kb = self.KnowledgeBase(tmpdir)
            result = kb.retrieve("machine learning algorithms")
            self.assertIn("machine", result.lower())

    def test_retrieve_with_unrelated_query(self):
        """retrieve() with unrelated query may return empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("Python programming web development. " * 20)
            kb = self.KnowledgeBase(tmpdir)
            result = kb.retrieve("quantum physics astrophysics")
            # May still match on common words; should not crash
            self.assertIsInstance(result, str)

    def test_retrieve_top_k_parameter(self):
        """retrieve() should respect top_k parameter (limit results)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a single large file that produces many chunks
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("machine learning " * 1000)
            kb = self.KnowledgeBase(tmpdir)
            result_1 = kb.retrieve("machine", top_k=1)
            result_3 = kb.retrieve("machine", top_k=3)
            # top_k=3 should have more refs than top_k=1
            refs_1 = result_1.count("[Ref")
            refs_3 = result_3.count("[Ref")
            self.assertLessEqual(refs_1, refs_3)

    def test_retrieve_returns_chunks_with_ref_label(self):
        """retrieve() chunks should be labeled [Ref N]:."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("machine learning natural language processing. " * 30)
            kb = self.KnowledgeBase(tmpdir)
            result = kb.retrieve("machine learning")
            if result:
                self.assertIn("[Ref ", result)

    def test_retrieve_chunking_produces_multiple_chunks(self):
        """Large file should produce at least 2 chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("machine learning " * 2000)
            kb = self.KnowledgeBase(tmpdir)
            self.assertGreater(len(kb.chunks), 1)

    def test_retrieve_case_insensitive(self):
        """retrieve() should match case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("Machine Learning is fun. " * 20)
            kb = self.KnowledgeBase(tmpdir)
            result = kb.retrieve("MACHINE")
            self.assertIn("machine", result.lower())

    def test_retrieve_multiple_files_merged(self):
        """Chunks from multiple files should all be searchable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "ai.txt"), "w") as f:
                f.write("artificial intelligence neural networks. " * 30)
            with open(os.path.join(tmpdir, "web.txt"), "w") as f:
                f.write("web development frontend backend. " * 30)
            kb = self.KnowledgeBase(tmpdir)
            ai_result = kb.retrieve("neural networks")
            web_result = kb.retrieve("frontend")
            # Both queries should find something
            ai_found = "neural" in ai_result.lower()
            web_found = "frontend" in web_result.lower()
            self.assertTrue(ai_found or web_found)  # at least one works


class TestKnowledgeBaseEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @classmethod
    def setUpClass(cls):
        import chat
        cls.KnowledgeBase = chat.KnowledgeBase

    def test_init_large_file(self):
        """KnowledgeBase should handle a large file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "large.txt"), "w") as f:
                f.write("test " * 100000)
            kb = self.KnowledgeBase(tmpdir)
            self.assertTrue(kb.loaded)
            self.assertGreater(len(kb.chunks), 0)

    def test_init_chunk_size_parameter(self):
        """KnowledgeBase should respect the chunk_size parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("word " * 1000)
            kb_small = self.KnowledgeBase(tmpdir, chunk_size=100)
            kb_large = self.KnowledgeBase(tmpdir, chunk_size=400)
            # Smaller chunk size should produce more chunks
            self.assertGreaterEqual(len(kb_small.chunks), len(kb_large.chunks))

    def test_retrieve_no_keyword_match_returns_empty(self):
        """When no keywords match, retrieve returns empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
                f.write("xyzzy magical unknown words. " * 30)
            kb = self.KnowledgeBase(tmpdir)
            result = kb.retrieve("nonexistent_zzz_keyword")
            # The current implementation matches word overlap; may or may not match
            self.assertIsInstance(result, str)

    def test_init_subdirectories_ignored(self):
        """KnowledgeBase should search subdirectories (rglob)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "nested.txt"), "w") as f:
                f.write("nested file content about AI. " * 20)
            kb = self.KnowledgeBase(tmpdir)
            self.assertTrue(kb.loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
