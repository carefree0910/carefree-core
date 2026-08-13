import pytest
import unittest

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing import Type
from pathlib import Path
from zipfile import ZipFile
from zipfile import BadZipFile
from tempfile import TemporaryDirectory
from unittest.mock import patch
from core.toolkit.misc import compress
from core.toolkit.misc import ISerializableDataClass
from core.toolkit.pipeline import get_folder
from core.toolkit.pipeline import check_requirement
from core.toolkit.pipeline import IBlock
from core.toolkit.pipeline import IPipeline


class TestPipeline(unittest.TestCase):
    def test_iblock(self):
        @dataclass
        class TestConfig(ISerializableDataClass["TestConfig"]):
            value: str = "test"

        TestConfig.d = {}
        TestConfig.register("test_config")(TestConfig)

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
                return TestConfig

            @property
            def block_base(self) -> Type:
                return TestBlock

        config = TestConfig()
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
        restored = TestPipeline.init(TestConfig("placeholder")).from_info(info)
        self.assertEqual(restored.config, config)
        self.assertEqual(len(restored.blocks), 2)


class TestBuildContracts(unittest.TestCase):
    def setUp(self):
        @dataclass
        class ContractConfig(ISerializableDataClass["ContractConfig"]):
            value: str = "initial"

        ContractConfig.d = {}
        ContractConfig.register("contract_config")(ContractConfig)

        class ContractBlock(IBlock):
            d = {}

            def build(self, config: Any) -> None:
                pass

        @ContractBlock.register("contract_first")
        class FirstBlock(ContractBlock):
            pass

        @ContractBlock.register("contract_second")
        class SecondBlock(ContractBlock):
            pass

        @ContractBlock.register("contract_third")
        class ThirdBlock(ContractBlock):
            pass

        class ContractPipeline(IPipeline):
            @classmethod
            def init(cls, config: Any) -> "ContractPipeline":
                self = cls()
                self.config = config
                return self

            @property
            def config_base(self) -> Type:
                return ContractConfig

            @property
            def block_base(self) -> Type:
                return ContractBlock

        self.config_type = ContractConfig
        self.pipeline_type = ContractPipeline
        self.first_type = FirstBlock
        self.second_type = SecondBlock
        self.third_type = ThirdBlock

    def test_requirement_choices_and_none(self):
        class Fallback(IBlock):
            __identifier__ = "fallback"

            def build(self, config: Any) -> None:
                pass

        class Alternative(IBlock):
            __identifier__ = "primary | fallback"

            def build(self, config: Any) -> None:
                pass

        class Optional(IBlock):
            __identifier__ = "missing | none"

            def build(self, config: Any) -> None:
                pass

        class Consumer(IBlock):
            __identifier__ = "consumer"

            @property
            def requirements(self):
                return [Alternative, Optional]

            def build(self, config: Any) -> None:
                pass

        check_requirement(Consumer(), {"fallback": Fallback()})
        with self.assertRaises(ValueError):
            check_requirement(Consumer(), {})

    def test_previous_only_contains_prior_blocks(self):
        pipeline = self.pipeline_type.init(self.config_type())
        first = self.first_type()
        second = self.second_type()
        third = self.third_type()

        pipeline.build(first, second)
        pipeline.build(third)

        self.assertEqual(first.previous, {})
        self.assertEqual(list(second.previous), ["contract_first"])
        self.assertIs(second.previous["contract_first"], first)
        self.assertEqual(
            list(third.previous),
            ["contract_first", "contract_second"],
        )
        self.assertIs(third.previous["contract_first"], first)
        self.assertIs(third.previous["contract_second"], second)
        self.assertEqual(first.previous, {})
        self.assertEqual(list(second.previous), ["contract_first"])

    def test_legacy_pipeline_info_roundtrip(self):
        legacy_info = {
            "blocks": ["contract_first", "contract_second"],
            "config": {
                "type": "contract_config",
                "info": {"value": "legacy"},
            },
        }
        original_info = deepcopy(legacy_info)

        pipeline = self.pipeline_type.init(self.config_type()).from_info(legacy_info)

        self.assertEqual(legacy_info, original_info)
        self.assertEqual(pipeline.config.value, "legacy")
        self.assertIsInstance(pipeline.blocks[0], self.first_type)
        self.assertIsInstance(pipeline.blocks[1], self.second_type)
        self.assertIs(
            pipeline.blocks[1].previous["contract_first"],
            pipeline.blocks[0],
        )
        self.assertEqual(pipeline.to_info(), original_info)

        duplicate_info = {
            "blocks": ["contract_first", "contract_first"],
            "config": original_info["config"],
        }
        with self.assertRaises(ValueError):
            self.pipeline_type.init(self.config_type()).from_info(duplicate_info)

    def test_build_preflights_all_requirements(self):
        class MutatingBlock(self.first_type):
            __identifier__ = "contract_mutating"
            was_built = False

            def build(self, config: Any) -> None:
                self.was_built = True
                config.value = "mutated"

        class MissingBlock(self.first_type):
            __identifier__ = "contract_missing"

        class NeedsMissingBlock(self.first_type):
            __identifier__ = "contract_needs_missing"

            @property
            def requirements(self):
                return [MissingBlock]

        events = []

        class TrackingPipeline(self.pipeline_type):
            def before_block_build(self, block: Any) -> None:
                events.append(("before", block.__identifier__))

            def after_block_build(self, block: Any) -> None:
                events.append(("after", block.__identifier__))

        pipeline = TrackingPipeline.init(self.config_type())
        baseline = self.first_type()
        pipeline.build(baseline)
        events.clear()
        mutating = MutatingBlock()

        with self.assertRaises(ValueError):
            pipeline.build(mutating, NeedsMissingBlock(), MissingBlock())

        self.assertFalse(mutating.was_built)
        self.assertEqual(events, [])
        self.assertEqual(pipeline.blocks, [baseline])
        self.assertEqual(pipeline.block_mappings, {"contract_first": baseline})
        self.assertEqual(pipeline.config.value, "initial")

    def test_build_rejects_duplicate_identifiers(self):
        class DuplicateBlock(self.first_type):
            __identifier__ = "contract_duplicate"

        pipeline = self.pipeline_type.init(self.config_type())
        with self.assertRaises(ValueError):
            pipeline.build(DuplicateBlock(), DuplicateBlock())
        self.assertEqual(pipeline.blocks, [])
        self.assertEqual(pipeline.block_mappings, {})

        pipeline = self.pipeline_type.init(self.config_type())
        original = DuplicateBlock()
        pipeline.build(original)
        with self.assertRaises(ValueError):
            pipeline.build(DuplicateBlock())
        self.assertEqual(pipeline.blocks, [original])
        self.assertIs(pipeline.get_block("contract_duplicate"), original)


class TestGetFolder(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_folder(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.test_dir = tmp_path / "source"
        self.test_dir.mkdir()
        self.test_file = self.test_dir / "test.txt"
        with self.test_file.open("w") as f:
            f.write("Hello, World!")
        self.zip_file = self.test_dir / "test.zip"
        with ZipFile(self.zip_file, "w") as zipf:
            zipf.write(self.test_file, "test.txt")

    def _tracked_temporary_directories(self):
        directories = []

        def make_temporary_directory(*args, **kwargs):
            kwargs["dir"] = self.tmp_path
            temporary_directory = TemporaryDirectory(*args, **kwargs)
            directories.append(Path(temporary_directory.name))
            return temporary_directory

        return directories, make_temporary_directory

    def test_existing_folder(self):
        with get_folder(self.test_dir) as folder:
            self.assertTrue(folder.is_dir())
            self.assertEqual(folder, self.test_dir)
        self.assertTrue(self.test_dir.is_dir())
        self.assertEqual(self.test_file.read_text(), "Hello, World!")

    def test_force_new_folder(self):
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with get_folder(self.test_dir, force_new=True) as folder:
                self.assertTrue(folder.is_dir())
                self.assertNotEqual(folder, self.test_dir)
                self.assertTrue((folder / "test.txt").is_file())
                copied_folder = folder
                (folder / "test.txt").write_text("changed")
                raise RuntimeError("body failed")
        self.assertFalse(copied_folder.exists())
        self.assertEqual(self.test_file.read_text(), "Hello, World!")

    def test_non_existent_folder(self):
        with self.assertRaises(ValueError):
            with get_folder(self.test_dir / "non_existent"):
                pass

    def test_zip_file(self):
        serializer_folder = self.tmp_path / "serializer"
        serializer_folder.mkdir()
        (serializer_folder / "info.json").write_text("{}")
        compress(serializer_folder)

        with get_folder(serializer_folder) as folder:
            self.assertTrue(folder.is_dir())
            self.assertEqual((folder / "info.json").read_text(), "{}")
            extracted_folder = folder
        self.assertFalse(extracted_folder.exists())
        self.assertTrue(Path(f"{serializer_folder}.zip").is_file())

    def test_owned_temporary_directories_are_cleaned_up_on_failures(self):
        def assert_cleanup(action, error_type):
            directories, temporary_directory = self._tracked_temporary_directories()
            with patch(
                "core.toolkit.pipeline.TemporaryDirectory",
                side_effect=temporary_directory,
            ):
                with self.assertRaises(error_type):
                    action()
            self.assertEqual(len(directories), 1)
            self.assertFalse(directories[0].exists())

        def copy_failure():
            with patch(
                "core.toolkit.pipeline.shutil.copytree",
                side_effect=RuntimeError("copy failed"),
            ):
                with get_folder(self.test_dir, force_new=True):
                    pass

        malformed = self.tmp_path / "malformed"
        Path(f"{malformed}.zip").write_bytes(b"not a zip")

        def malformed_failure():
            with get_folder(malformed):
                pass

        def extract_failure():
            with patch.object(
                ZipFile,
                "extractall",
                side_effect=RuntimeError("extract failed"),
            ):
                with get_folder(self.test_dir / "test"):
                    pass

        assert_cleanup(copy_failure, RuntimeError)
        assert_cleanup(malformed_failure, BadZipFile)
        assert_cleanup(extract_failure, RuntimeError)


class TestCheckRequirement(unittest.TestCase):
    def setUp(self):
        class Requirement1(IBlock):
            __identifier__ = "req1"

            def build(self, config: Any) -> None:
                pass

        class Requirement2(IBlock):
            __identifier__ = "req2"

            def build(self, config: Any) -> None:
                pass

        class BlockWithRequirements(IBlock):
            __identifier__ = "block_with_requirements"

            @property
            def requirements(self):
                return [Requirement1, Requirement2]

            def build(self, config: Any) -> None:
                pass

        class BlockWithoutRequirements(IBlock):
            __identifier__ = "block_without_requirements"

            def build(self, config: Any) -> None:
                pass

        self.block_with_requirements = BlockWithRequirements()
        self.block_without_requirements = BlockWithoutRequirements()
        self.previous_blocks = {
            "req1": Requirement1(),
            "req2": Requirement2(),
        }

    def test_requirements_met(self):
        check_requirement(self.block_with_requirements, self.previous_blocks)

    def test_requirements_not_met(self):
        with self.assertRaises(ValueError):
            check_requirement(
                self.block_with_requirements,
                {"req1": self.previous_blocks["req1"]},
            )

    def test_no_requirements(self):
        check_requirement(self.block_without_requirements, {})


if __name__ == "__main__":
    unittest.main()
