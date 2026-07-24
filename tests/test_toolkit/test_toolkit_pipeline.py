import pytest
import unittest

from typing import Any
from typing import Type
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import Mock
from core.toolkit.pipeline import get_folder
from core.toolkit.pipeline import check_requirement
from core.toolkit.pipeline import IBlock
from core.toolkit.pipeline import IPipeline


class TestPipeline(unittest.TestCase):
    def test_iblock(self):
        @IBlock.register("test_block")
        class TestBlock(IBlock):
            def build(self, config: Any) -> None:
                pass

        @IBlock.register("test_block2")
        class TestBlock2(IBlock):
            def build(self, config: Any) -> None:
                pass

        class TestPipeline(IPipeline):
            @classmethod
            def init(cls, config: Any) -> "TestPipeline":
                self = cls()
                self.config = config
                return self

            @property
            def config_base(self) -> Type:
                return Mock()

            @property
            def block_base(self) -> Type:
                return TestBlock

        config = Mock()
        config.to_pack.return_value.asdict.return_value = {"type": "test"}
        p = TestPipeline.init(config)
        first = TestBlock()
        second = TestBlock2()
        p.build(first, second)
        with self.assertRaises(ValueError):
            p.get_block(TestBlock).get_previous(TestBlock)
        self.assertIs(p.get_block(TestBlock2).get_previous(TestBlock), first)
        with self.assertRaises(ValueError):
            p.get_block("missing")

        info = p.to_info()
        restored = TestPipeline.init(Mock()).from_info(info)
        self.assertEqual(len(restored.blocks), 2)


class TestGetFolder(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_folder(self, tmp_path: Path) -> None:
        self.test_dir = tmp_path / "source"
        self.test_dir.mkdir()
        self.test_file = self.test_dir / "test.txt"
        with self.test_file.open("w") as f:
            f.write("Hello, World!")
        self.zip_file = self.test_dir / "test.zip"
        with ZipFile(self.zip_file, "w") as zipf:
            zipf.write(self.test_file, "test.txt")

    def test_existing_folder(self):
        with get_folder(self.test_dir) as folder:
            self.assertTrue(folder.is_dir())
            self.assertEqual(folder, self.test_dir)

    def test_force_new_folder(self):
        with get_folder(self.test_dir, force_new=True) as folder:
            self.assertTrue(folder.is_dir())
            self.assertNotEqual(folder, self.test_dir)
            self.assertTrue((folder / "test.txt").is_file())

    def test_non_existent_folder(self):
        with self.assertRaises(ValueError):
            with get_folder(self.test_dir / "non_existent"):
                pass

    def test_zip_file(self):
        with get_folder(self.test_dir / "test") as folder:
            self.assertTrue(folder.is_dir())
            self.assertTrue((folder / "test.txt").is_file())


class TestCheckRequirement(unittest.TestCase):
    def setUp(self):
        req1 = Mock()
        req2 = Mock()
        req1.__identifier__ = "req1"
        req2.__identifier__ = "req2"

        self.block_with_requirements = Mock()
        self.block_with_requirements.__identifier__ = "block_with_requirements"
        self.block_with_requirements.requirements = [req1, req2]

        self.block_without_requirements = Mock()
        self.block_without_requirements.__identifier__ = "block_without_requirements"
        self.block_without_requirements.requirements = []

        self.previous_blocks = {"req1": Mock(), "req2": Mock()}

    def test_requirements_met(self):
        check_requirement(self.block_with_requirements, self.previous_blocks)

    def test_requirements_not_met(self):
        with self.assertRaises(ValueError):
            check_requirement(self.block_with_requirements, {"req1": Mock()})

    def test_no_requirements(self):
        check_requirement(self.block_without_requirements, {})


if __name__ == "__main__":
    unittest.main()
