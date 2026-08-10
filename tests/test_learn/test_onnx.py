import torch
import pytest
import unittest

import numpy as np
import core.learn as cflearn

from typing import List
from typing import Optional
from pathlib import Path
from unittest.mock import patch
from core.learn.schema import DataLoader
from core.toolkit.types import np_dict_type


class TestONNX(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    @staticmethod
    def _make_model(input_dim: int, output_dim: int) -> cflearn.IModel:
        @cflearn.IModel.register("$onnx_state_contract", allow_duplicate=True)
        class StateContractModel(cflearn.CommonModel):
            auxiliary: torch.nn.BatchNorm1d

            @property
            def all_modules(self) -> List[torch.nn.Module]:
                return [self.m, self.auxiliary, self.loss]

            def build(self, config: cflearn.Config) -> None:
                super().build(config)
                self.auxiliary = torch.nn.BatchNorm1d(output_dim)

        return cflearn.IModel.from_config(
            cflearn.Config(
                model=StateContractModel.__identifier__,
                module_name="fcnn",
                module_config={
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "hidden_units": [4],
                    "batch_norm": True,
                },
                loss_name="mse",
            )
        )

    @staticmethod
    def _module_map(model):
        return {
            id(child): child for root in model.all_modules for child in root.modules()
        }

    @staticmethod
    def _tensor_map(modules):
        return {
            id(tensor): tensor
            for module in modules.values()
            for tensor in [
                *module.parameters(recurse=False),
                *module.buffers(recurse=False),
            ]
        }

    def _snapshot(self, model):
        modules = self._module_map(model)
        tensors = self._tensor_map(modules)
        return (
            model.all_modules,
            [(module, module.training) for module in modules.values()],
            [
                (tensor, tensor.detach().clone(), tensor.requires_grad)
                for tensor in tensors.values()
            ],
        )

    def _assert_snapshot(self, model, snapshot) -> None:
        expected_roots, expected_modules, expected_tensors = snapshot
        roots = model.all_modules
        self.assertEqual(len(roots), len(expected_roots))
        for root, expected_root in zip(roots, expected_roots):
            self.assertIs(root, expected_root)
        modules = self._module_map(model)
        self.assertSetEqual(
            set(modules),
            {id(module) for module, _ in expected_modules},
        )
        for module, expected_mode in expected_modules:
            self.assertEqual(module.training, expected_mode)
        tensors = self._tensor_map(modules)
        self.assertSetEqual(
            set(tensors),
            {id(tensor) for tensor, _, _ in expected_tensors},
        )
        for tensor, expected_value, expected_requires_grad in expected_tensors:
            self.assertIs(tensors[id(tensor)], tensor)
            self.assertEqual(tensor.device, expected_value.device)
            self.assertTrue(torch.equal(tensor, expected_value))
            self.assertEqual(tensor.requires_grad, expected_requires_grad)

    @staticmethod
    def _prepare_export_contract(model) -> None:
        model.m.train()
        next(iter(model.m.children())).eval()
        model.auxiliary.eval()
        model.loss.train()
        with torch.no_grad():
            next(model.m.parameters()).view(-1)[0] = 1.0e-40
            buffer = next(b for b in model.m.buffers() if b.is_floating_point())
            buffer.view(-1)[0] = 1.0e-40

    def test_onnx(self) -> None:
        input_dim = 11
        output_dim = 7
        num_samples = 123
        batch_size = 17

        x = np.random.randn(num_samples, input_dim).astype(np.float32)
        y = np.random.randn(num_samples, output_dim).astype(np.float32)
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = batch_size
        loader = data.build_loader(x, y)

        model = self._make_model(input_dim, output_dim)
        model_inference = cflearn.Inference(model=model)
        model_outputs = model_inference.get_outputs(loader).forward_results

        onnx_file = self.tmp_path / "test.onnx"
        snapshot = self._snapshot(model)
        exported = model.to_onnx(str(onnx_file), loader.get_input_sample())
        self.assertIs(exported, model)
        self._assert_snapshot(model, snapshot)
        exported = model.to_onnx(
            str(onnx_file),
            loader.get_input_sample(),
            dynamic_axes=[0],
            simplify=False,
            forward_fn=lambda d: model.onnx_forward(d)[cflearn.PREDICTIONS_KEY],
            num_samples=1,
        )
        self.assertIs(exported, model)
        self._assert_snapshot(model, snapshot)

        @cflearn.IMetric.register("foo", allow_duplicate=True)
        class FooMetric(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return True

            @property
            def requires_all(self) -> bool:
                return True

            def forward(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[DataLoader] = None,
            ) -> float:
                return 0.12

        onnx_inference = cflearn.Inference(onnx=str(onnx_file))
        onnx_inference.get_outputs(loader, metrics=FooMetric(), return_labels=True)
        num_injections = 0

        def inject_outputs(_, __) -> None:
            nonlocal num_injections
            num_injections += 1

        onnx_outputs = onnx_inference.get_outputs(
            loader,
            inject_outputs_fn=inject_outputs,
        ).forward_results
        self.assertEqual(num_injections, len(loader))

        for k in model_outputs:
            mk_out = model_outputs[k]
            ok_out = onnx_outputs[k]
            np.testing.assert_array_almost_equal(mk_out, ok_out)

        empty_loader = data.build_loader(x[:0], y[:0])
        self.assertIsNotNone(onnx_inference.onnx)
        with patch.object(
            onnx_inference.onnx,
            "predict",
            wraps=onnx_inference.onnx.predict,
        ) as predict:
            empty_outputs = onnx_inference.get_outputs(empty_loader)
        predict.assert_not_called()
        self.assertDictEqual(empty_outputs.forward_results, {})
        self.assertDictEqual(empty_outputs.labels, {})
        self.assertIsNone(empty_outputs.metric_outputs)
        self.assertIsNone(empty_outputs.loss_items)

    def test_to_moves_all_modules(self) -> None:
        model = self._make_model(3, 2)

        self.assertIs(model.to("meta"), model)
        self.assertEqual(model.device.type, "meta")

        for module in model.all_modules:
            for tensor in [*module.parameters(), *module.buffers()]:
                self.assertEqual(tensor.device.type, "meta")

    def test_eval_context_preserves_unified_mode(self) -> None:
        model = self._make_model(3, 2)
        modules = list(self._module_map(model).values())

        for training in [True, False]:
            for module in model.all_modules:
                module.train(training)
            with model.eval_context(use_inference=False):
                self.assertTrue(all(not module.training for module in modules))
            self.assertTrue(all(module.training is training for module in modules))

    def test_state_dict_keeps_main_module_payload(self) -> None:
        model = self._make_model(3, 2)
        main_states = {
            key: value.detach().clone() for key, value in model.m.state_dict().items()
        }
        auxiliary_states = {
            key: value.detach().clone()
            for key, value in model.auxiliary.state_dict().items()
        }

        states = model.state_dict()

        self.assertListEqual(list(states), list(main_states))
        torch.testing.assert_close(states, main_states)

        loaded_states = {
            key: torch.zeros_like(value) for key, value in main_states.items()
        }
        self.assertIsNone(model.load_state_dict(loaded_states))
        torch.testing.assert_close(model.m.state_dict(), loaded_states)
        torch.testing.assert_close(
            model.auxiliary.state_dict(),
            auxiliary_states,
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="P0-09: preserve model state after ONNX export",
    )
    def test_onnx_export_preserves_model_state(self) -> None:
        model = self._make_model(3, 2)
        self._prepare_export_contract(model)
        snapshot = self._snapshot(model)
        onnx_file = self.tmp_path / "state_contract.onnx"

        exported = model.to_onnx(
            str(onnx_file),
            {cflearn.INPUT_KEY: torch.randn(2, 3)},
            simplify=False,
            verbose=False,
        )

        self.assertIs(exported, model)
        self.assertTrue(onnx_file.is_file())
        self._assert_snapshot(model, snapshot)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="P0-09: restore model state after ONNX export failure",
    )
    def test_onnx_export_failure_restores_model_state(self) -> None:
        model = self._make_model(3, 2)
        self._prepare_export_contract(model)
        snapshot = self._snapshot(model)

        def fail_forward(_):
            raise RuntimeError("intentional ONNX export failure")

        with self.assertRaisesRegex(Exception, "intentional ONNX export failure"):
            model.to_onnx(
                str(self.tmp_path / "failed_state_contract.onnx"),
                {cflearn.INPUT_KEY: torch.randn(2, 3)},
                forward_fn=fail_forward,
                output_names=[cflearn.PREDICTIONS_KEY],
                simplify=False,
                verbose=False,
            )
        self._assert_snapshot(model, snapshot)


if __name__ == "__main__":
    unittest.main()
