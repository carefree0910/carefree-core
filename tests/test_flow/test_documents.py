import unittest
import core.flow as cflow


class TestDocuments(unittest.TestCase):
    def test_generate_documents(self):
        with self.assertRaises(ValueError):
            cflow.generate_documents("docs.txt")
        cflow.generate_documents("docs.md")
        cflow.generate_documents("docs_rag.md", rag=True)


if __name__ == "__main__":
    unittest.main()
