import tempfile
import unittest

from core import flow as cflow
from typing import Any
from pathlib import Path
from core.flow.docs import generate_document


class TestDocuments(unittest.TestCase):
    def test_generate_document(self):
        @cflow.Node.register("foo", allow_duplicate=True)
        class Foo(cflow.Node):
            async def execute(self) -> Any:
                pass

        @cflow.Node.register("bar", allow_duplicate=True)
        class Bar(cflow.Node):
            @classmethod
            def get_schema(cls) -> cflow.Schema:
                return cflow.Schema(input_names=["foo"], output_names=["bar"])

            async def execute(self) -> Any:
                pass

        generate_document(Foo, False)
        generate_document(Bar, False)

    def test_generate_documents(self):
        with self.assertRaises(ValueError):
            cflow.generate_documents("docs.txt")
        cflow.generate_documents("docs.md")
        cflow.generate_documents("docs_rag.md", rag=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflows_root = root / "workflows"
            workflows_root.mkdir()
            (root / "foo.py").touch()
            bar_json = workflows_root / "bar.json"
            with bar_json.open("w") as f:
                f.write("{}")
            cflow.generate_documents("docs.md", examples_root=root)
            cflow.generate_documents("docs_rag.md", rag=True, examples_root=root)


if __name__ == "__main__":
    unittest.main()
