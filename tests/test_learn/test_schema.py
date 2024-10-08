import os
import torch
import tempfile
import unittest

import numpy as np
import core.learn as cflearn

from enum import Enum
from typing import Any
from typing import Dict
from core.learn.schema import *


class TestSchema(unittest.TestCase):
    def test_functions(self):
        class Foo(Enum):
            A = 1
            B = 2

        sws = np.random.random([3, 5]), np.random.random([3, 5])
        tsw, vsw = split_sw(sws)
        np.testing.assert_allclose(tsw, norm_sw(sws[0]))
        np.testing.assert_allclose(vsw, norm_sw(sws[1]))
        self.assertFalse(check_data_is_info(Foo.A))
        cfg = cflearn.TrainerConfig()
        self.assertEqual(weighted_loss_score(cfg, {}), 0.0)
        self.assertEqual(weighted_loss_score(cfg, {"foo": 1.0, "bar": 2.0}), -1.5)
        cfg.loss_metrics_weights = dict(foo=1.0, bar=2.0)
        self.assertEqual(weighted_loss_score(cfg, {}), 0.0)
        self.assertEqual(weighted_loss_score(cfg, {"foo": 1.0, "bar": 2.0}), -5)

    def test_data_loader(self):
        class MockArray:
            def __init__(self, value):
                self.value = value

            def __len__(self):
                return 1

            def __getitem__(self, item):
                return self.value

        data = cflearn.testing.linear_data(3)[0]
        x, y = data.bundle.x_train, data.bundle.y_train
        loader = data.build_loader(x, y)
        loader.get_full_batch("cpu")
        x = dict(a=MockArray(0), b=MockArray([]))
        data = cflearn.ArrayDictData.init().fit(x)
        loader = data.build_loaders()[0]
        loader.get_input_sample()
        loader.dataset[0]
        data = cflearn.ArrayDictData.init()
        with self.assertRaises(RuntimeError):
            data.build_loaders()
        data.is_ready = True
        with self.assertRaises(RuntimeError):
            data.build_loaders()
        data.from_npd(dict(x=dict(a=MockArray(0))))
        data_config = cflearn.DataConfig(valid_loader_configs=dict(num_workers=7))
        data = cflearn.ArrayDictData.init(data_config).fit(x, x_valid=x)
        loader = data.to_loader(
            data.to_datasets(cflearn.DataBundle(x, x_valid=x), for_inference=False)[1],
            shuffle=False,
            batch_size=1,
            for_inference=False,
            is_validation=True,
        )
        self.assertEqual(loader.num_workers, 7)

    def test_data_config(self):
        config = cflearn.DataConfig()

        @cflearn.IDataBlock.register("foo", allow_duplicate=True)
        class FooDataBlock(cflearn.IDataBlock):
            def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
                return bundle

            def fit_transform(self, bundle: DataBundle) -> DataBundle:
                return bundle

            def to_info(self) -> Dict[str, Any]:
                return {}

        config.add_blocks(FooDataBlock)
        self.assertListEqual(config.block_names, ["foo"])
        config.add_blocks(FooDataBlock)
        self.assertListEqual(config.block_names, ["foo"])
        config.set_blocks(FooDataBlock, FooDataBlock)
        self.assertListEqual(config.block_names, ["foo"])
        foo_config = dict(a=1, b=2)
        config.block_configs = dict(foo=foo_config)
        foo_block = FooDataBlock()
        foo_block.build(config)
        self.assertTrue(foo_block.is_local_rank_0)
        self.assertDictEqual(foo_block.configs, foo_config)

    def test_data_bundle(self):
        bundle = cflearn.DataBundle("foo")
        info = bundle.to_info()
        self.assertDictEqual(info, dict(x_train="foo"))
        self.assertDictEqual(info, cflearn.DataBundle.empty().from_info(info).to_info())
        bundle = cflearn.DataBundle(np.random.random([13, 31]))
        loaded = cflearn.DataBundle.empty().from_npd(bundle.to_npd())
        np.testing.assert_allclose(bundle.x_train, loaded.x_train)
        bundle = cflearn.DataBundle(torch.randn(13, 31))
        loaded = cflearn.DataBundle.empty().from_npd(bundle.to_npd())
        torch.testing.assert_close(bundle.x_train, loaded.x_train)
        bundle = cflearn.DataBundle(
            dict(
                a=dict(b=np.random.random([13, 31]), c=torch.randn(13, 31)),
                d=np.random.random([13, 31]),
                e=torch.randn(13, 31),
            ),
            "foo",
        )
        loaded = cflearn.DataBundle.empty().from_npd(bundle.to_npd())
        np.testing.assert_allclose(bundle.x_train["a"]["b"], loaded.x_train["a"]["b"])
        np.testing.assert_allclose(bundle.x_train["d"], loaded.x_train["d"])
        torch.testing.assert_close(bundle.x_train["a"]["c"], loaded.x_train["a"]["c"])
        torch.testing.assert_close(bundle.x_train["e"], loaded.x_train["e"])

        data = cflearn.testing.linear_data(3)[0]
        info = data.to_info()
        info["bundle"] = bundle.to_info()
        data.from_info(info)
        self.assertDictEqual(data.bundle.to_info(), dict(y_train="foo"))
        data.bundle = bundle
        loaded = data.from_npd(data.to_npd()).bundle
        np.testing.assert_allclose(bundle.x_train["a"]["b"], loaded.x_train["a"]["b"])
        np.testing.assert_allclose(bundle.x_train["d"], loaded.x_train["d"])
        torch.testing.assert_close(bundle.x_train["a"]["c"], loaded.x_train["a"]["c"])
        torch.testing.assert_close(bundle.x_train["e"], loaded.x_train["e"])

    def test_model(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="fcnn",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        m = cflearn.IModel.from_config(config)
        x, y = data.bundle.x_train, data.bundle.y_train
        loader = data.build_loader(x, y)
        inference = cflearn.Inference(model=m)
        o0 = inference.get_outputs(loader).forward_results[cflearn.PREDICTIONS_KEY]
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "model.pt")
            m.save(path)
            loaded = cflearn.IModel.load(path)
            inference = cflearn.Inference(model=loaded)
            o1 = inference.get_outputs(loader).forward_results[cflearn.PREDICTIONS_KEY]
            np.testing.assert_allclose(o0, o1)

    def test_config(self):
        with self.assertRaises(ValueError):
            cflearn.Config(mixed_precision=1)
        cflearn.Config(module_name="foo").sanity_check()
        with self.assertRaises(ValueError):
            cflearn.Config().sanity_check()
        with self.assertRaises(ValueError):
            cflearn.Config(mixed_precision=1)
        self.assertTrue(cflearn.Config().to_debug().is_debug)
        self.assertEqual(cflearn.Config().trainer_config, cflearn.TrainerConfig())
        config = cflearn.Config(mixed_precision=cflearn.PrecisionType.FP16)
        self.assertEqual(config.mixed_precision, "fp16")

    def test_trainer_state(self):
        state = TrainerState(
            num_epoch=1,
            batch_size=2,
            loader_length=10,
            min_num_sample=9,
            enable_logging=False,
            min_snapshot_epoch_gap=2,
            manual_snapshot_epochs=[1],
        )
        self.assertEqual(state.snapshot_start_step, 5)
        self.assertEqual(state.num_step_per_snapshot, 5)
        self.assertSetEqual(state.manual_snapshot_steps, {10})
        self.assertFalse(state.should_terminate)
        self.assertFalse(state.should_log_losses)
        self.assertFalse(state.should_log_metrics_msg)
        self.assertFalse(state.can_snapshot)
        state.epoch = -1
        self.assertTrue(state.can_snapshot)


if __name__ == "__main__":
    unittest.main()
