import sys
import pytest
import unittest
import importlib

from core.learn.toolkit import *
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from core.toolkit.misc import random_hash
from safetensors.torch import save_file


class TestToolkit(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def test_new_seed(self) -> None:
        for _ in range(10):
            seed = new_seed()
            self.assertLessEqual(seed, max_seed_value)
            self.assertGreaterEqual(seed, min_seed_value)

    def test_optional_onnxruntime_import(self) -> None:
        toolkit = importlib.import_module("core.learn.toolkit")
        spec = importlib.util.spec_from_file_location(
            "core.learn._toolkit_without_onnxruntime",
            toolkit.__file__,
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"onnxruntime": None}):
            spec.loader.exec_module(module)
        self.assertIsNone(module.InferenceSession)
        with self.assertRaisesRegex(
            ValueError,
            r"pip install 'carefree-core\[onnx\]'",
        ):
            module.ONNX("model.onnx")

    def test_env_workspace(self) -> None:
        env_workspace = "env_workspace"
        set_environ_workspace(env_workspace)
        self.assertEqual(get_environ_workspace(), env_workspace)
        unset_environ_workspace()
        self.assertIsNone(get_environ_workspace())

    def test_seed_everything(self) -> None:
        seed = 123
        shape = 3, 5, 11
        seed_everything(seed)
        a0 = torch.randn(*shape)
        a1 = torch.randn(*shape)
        seed_everything(seed)
        a2 = torch.randn(*shape)
        seed_everything(-1)
        a3 = torch.randn(*shape)
        seed_everything(-1)
        a4 = torch.randn(*shape)
        torch.testing.assert_close(a0, a2)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(a0, a1)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(a3, a4)

    def test_get_torch_device(self) -> None:
        cpu = torch.device("cpu")
        cuda1 = torch.device("cuda:1")
        self.assertTrue(get_torch_device(None) == cpu)
        self.assertTrue(get_torch_device("cpu") == cpu)
        with patch(
            "core.learn.toolkit.torch.device", return_value=cuda1
        ) as mock_device:
            self.assertTrue(get_torch_device(1) == cuda1)
            self.assertTrue(get_torch_device("1") == cuda1)
        self.assertEqual(
            [mock_call.args for mock_call in mock_device.call_args_list],
            [(1,), (1,)],
        )
        self.assertTrue(get_torch_device("cuda:1") == cuda1)
        self.assertTrue(get_torch_device(cuda1) == cuda1)

    def test_seed_within_range(self) -> None:
        seed = 42
        with patch("core.learn.toolkit.random.seed") as mock_random_seed, patch(
            "core.learn.toolkit.np.random.seed"
        ) as mock_np_seed, patch(
            "core.learn.toolkit.torch.manual_seed"
        ) as mock_torch_seed, patch(
            "core.learn.toolkit.torch.cuda.manual_seed_all"
        ) as mock_cuda_seed_all:
            result = seed_everything(seed)
        self.assertEqual(result, seed)
        mock_random_seed.assert_called_once_with(seed)
        mock_np_seed.assert_called_once_with(seed)
        mock_torch_seed.assert_called_once_with(seed)
        mock_cuda_seed_all.assert_called_once_with(seed)

    def test_seed_outside_range(self) -> None:
        for seed in [min_seed_value - 1, max_seed_value + 1]:
            new_seed = 42
            with patch(
                "core.learn.toolkit.new_seed", return_value=new_seed
            ) as mock_new_seed, patch(
                "core.learn.toolkit.random.seed"
            ) as mock_random_seed, patch(
                "core.learn.toolkit.np.random.seed"
            ) as mock_np_seed, patch(
                "core.learn.toolkit.torch.manual_seed"
            ) as mock_torch_seed, patch(
                "core.learn.toolkit.torch.cuda.manual_seed_all"
            ) as mock_cuda_seed_all:
                result = seed_everything(seed)
            self.assertEqual(result, new_seed)
            mock_new_seed.assert_called_once()
            mock_random_seed.assert_called_once_with(new_seed)
            mock_np_seed.assert_called_once_with(new_seed)
            mock_torch_seed.assert_called_once_with(new_seed)
            mock_cuda_seed_all.assert_called_once_with(new_seed)

    @patch("core.learn.toolkit.plt.show")
    def test_show_or_save(self, mock_show) -> None:
        show_or_save(None)
        mock_show.assert_called_once()

        fig = plt.figure()
        export_path = self.tmp_path / "figure.png"
        with patch.object(fig, "savefig", wraps=fig.savefig) as savefig:
            show_or_save(str(export_path), fig=fig)
        savefig.assert_called_once_with(str(export_path))
        self.assertTrue(export_path.is_file())

    @patch("core.learn.toolkit.plt.show")
    def test_show_or_return(self, mock_show) -> None:
        plt.figure()
        self.assertIsNone(show_or_return(False))
        mock_show.assert_called_once()
        self.assertIsInstance(show_or_return(True), np.ndarray)
        plt.close()

    def test_summary_restores_hooks_after_forward_error(self) -> None:
        model = nn.Sequential(nn.Linear(2, 1))
        caller_handle = model[0].register_forward_hook(lambda *_: None)
        expected_hooks = {
            module: tuple(module._forward_hooks.items()) for module in model.modules()
        }
        forward_error = RuntimeError("incompatible input shape")

        def fail_forward(_):
            raise forward_error

        try:
            with self.assertRaises(RuntimeError) as context:
                summary(model, {}, return_only=True, summary_forward=fail_forward)
            self.assertIs(context.exception, forward_error)
            for module, expected in expected_hooks.items():
                self.assertTupleEqual(tuple(module._forward_hooks.items()), expected)

            result = summary(
                model,
                {"input": torch.randn(1, 2)},
                return_only=True,
                summary_forward=lambda batch: model(batch["input"]),
            )
            self.assertIsInstance(result, str)
            for module, expected in expected_hooks.items():
                self.assertTupleEqual(tuple(module._forward_hooks.items()), expected)
        finally:
            caller_handle.remove()

    def test_get_tensors_from_safetensors(self) -> None:
        path = self.tmp_path / "example.safetensors"
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        save_file(expected_tensors, path)

        tensors = get_tensors(path)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)

    def test_get_tensors_from_pt(self) -> None:
        file_path = self.tmp_path / "example.pt"
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        torch.save(expected_tensors, file_path)

        tensors = get_tensors(file_path)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)

    def test_get_tensors_from_state_dict(self) -> None:
        state_dict = {
            "state_dict": {
                "tensor1": torch.tensor([1, 2, 3]),
                "tensor2": torch.tensor([4, 5, 6]),
            }
        }
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }

        tensors = get_tensors(state_dict)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)

    def test_get_tensors_from_dict(self) -> None:
        tensor_dict = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }

        tensors = get_tensors(tensor_dict)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)

    def test_get_tensors_from_number(self) -> None:
        with self.assertRaises(ValueError):
            get_tensors(1)

    def test_get_dtype(self) -> None:
        class Foo(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = nn.Parameter(torch.tensor([1.0]))

        foo = Foo()
        for dtype in [torch.float16, torch.float32]:
            foo.p.data = foo.p.data.to(dtype)
            self.assertEqual(get_dtype(foo), dtype)

        class Foo(nn.Module):
            pass

        foo = Foo()
        self.assertEqual(get_dtype(foo), torch.float32)

    def test_get_clones(self) -> None:
        m = nn.Linear(10, 2)
        clones = get_clones(m, 3, return_list=True)
        self.assertIsInstance(clones, list)
        self.assertEqual(len(clones), 3)
        for clone in clones:
            self.assertIsInstance(clone, nn.Linear)

    def test_empty_cuda_cache(self) -> None:
        empty_cuda_cache(None)
        empty_cuda_cache("cpu")

    def test_is_cpu(self) -> None:
        self.assertTrue(is_cpu(None))
        self.assertTrue(is_cpu("cpu"))
        self.assertTrue(is_cpu(torch.device("cpu")))
        self.assertFalse(is_cpu("cuda"))
        self.assertFalse(is_cpu("cuda:0"))
        self.assertFalse(is_cpu(torch.device("cuda")))
        self.assertFalse(is_cpu(torch.device("cuda:0")))

    def test_safe_clip_within_range(self):
        net = torch.tensor([1.0, 2.0, 3.0])
        safe_clip_(net)
        torch.testing.assert_close(net, torch.tensor([1.0, 2.0, 3.0]))

    def test_safe_clip_outside_range(self):
        finfo = torch.finfo(torch.float32)
        net = torch.tensor([finfo.min - 1, 0.0, finfo.max + 1])
        safe_clip_(net)
        torch.testing.assert_close(net, torch.tensor([finfo.min, 0.0, finfo.max]))

    def test_insert_intermediate_dims_with_more_dims(self):
        net = torch.tensor([[1.0, 2.0, 3.0]])
        ref = torch.tensor([[[1.0, 2.0, 3.0]]])
        result = insert_intermediate_dims(net, ref)
        expected = torch.tensor([[[1.0, 2.0, 3.0]]])
        torch.testing.assert_close(result, expected)
        net = np.array([[1.0, 2.0, 3.0]])
        ref = np.array([[[1.0, 2.0, 3.0]]])
        result = insert_intermediate_dims(net, ref)
        expected = np.array([[[1.0, 2.0, 3.0]]])
        np.testing.assert_allclose(result, expected)

    def test_insert_intermediate_dims_with_same_dims(self):
        net = torch.tensor([[1.0, 2.0, 3.0]])
        ref = torch.tensor([[1.0, 2.0, 3.0]])
        result = insert_intermediate_dims(net, ref)
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(result, expected)

    def test_insert_intermediate_dims_with_less_dims(self):
        net = torch.tensor([1.0, 2.0, 3.0])
        ref = torch.tensor([[[1.0, 2.0, 3.0]]])
        with self.assertRaises(ValueError):
            insert_intermediate_dims(net, ref)

    def test_fix_denormal_states(self) -> None:
        states = {
            "a": torch.tensor([1.0, 2.0, 1.0e-33]),
            "b": torch.tensor([4.0, 5.0, 6.0]),
            "c": torch.tensor([7, 8, 9], dtype=torch.int32),
        }
        expected_states = {
            "a": torch.tensor([1.0, 2.0, 0.0]),
            "b": torch.tensor([4.0, 5.0, 6.0]),
            "c": torch.tensor([7, 8, 9], dtype=torch.int32),
        }

        new_states = fix_denormal_states(states)

        self.assertIsInstance(new_states, dict)
        torch.testing.assert_close(new_states, expected_states)

    def test_fix_denormal_states_preserves_inputs(self) -> None:
        states = {
            "a": torch.tensor([1.0, 2.0, 1.0e-33]),
            "b": torch.tensor([4.0, 5.0, 6.0]),
            "c": torch.tensor([7, 8, 9], dtype=torch.int32),
        }
        original_states = {key: value.detach().clone() for key, value in states.items()}

        new_states = fix_denormal_states(states)

        self.assertIsNot(new_states, states)
        self.assertIsNot(new_states["a"], states["a"])
        self.assertIs(new_states["b"], states["b"])
        self.assertIs(new_states["c"], states["c"])
        torch.testing.assert_close(states, original_states, rtol=0.0, atol=0.0)

    def test_has_batch_norms_with_batch_norm_layers(self) -> None:
        m = nn.Sequential(nn.Linear(10, 2), nn.BatchNorm1d(2))

        result = has_batch_norms(m)

        self.assertTrue(result)

    def test_has_batch_norms_without_batch_norm_layers(self) -> None:
        m = nn.Sequential(nn.Linear(10, 2), nn.ReLU())

        result = has_batch_norms(m)

        self.assertFalse(result)

    def test_inject_parameters(self):
        src = nn.Linear(10, 2)
        tgt = nn.Linear(10, 2)
        inject_parameters(src, tgt)
        for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
            torch.testing.assert_close(src_param, tgt_param)
        tgt = nn.Linear(10, 2)
        exclude = {"bias"}
        inject_parameters(
            src,
            tgt,
            src_filter_fn=lambda k: k not in exclude,
            tgt_filter_fn=lambda k: k not in exclude,
        )
        torch.testing.assert_close(src.weight, tgt.weight)
        self.assertFalse(torch.allclose(src.bias, tgt.bias))
        inject_parameters(
            src,
            tgt,
            states_callback=lambda d: {k: torch.zeros_like(v) for k, v in d.items()},
        )
        for tgt_param in tgt.parameters():
            torch.testing.assert_close(tgt_param, torch.zeros_like(tgt_param))
        tgt = nn.Linear(10, 3)
        with self.assertRaises(ValueError):
            inject_parameters(src, tgt)

        class CustomLinear(nn.Module):
            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.bias = nn.Parameter(torch.randn(out_dim))
                self.custom_weight = nn.Parameter(torch.randn(out_dim, in_dim))

        tgt = CustomLinear(10, 2)
        with self.assertRaises(ValueError):
            inject_parameters(src, tgt)
        inject_parameters(src, tgt, custom_mappings={"weight": "custom_weight"})
        torch.testing.assert_close(src.weight, tgt.custom_weight)
        torch.testing.assert_close(src.bias, tgt.bias)

        tgt = nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 2))
        with self.assertRaises(ValueError):
            inject_parameters(src, tgt)

    def test_sorted_param_diffs(self) -> None:
        m1 = nn.Linear(10, 2)
        m2 = nn.Linear(10, 2)

        diffs = sorted_param_diffs(m1, m2)

        self.assertIsInstance(diffs, Diffs)
        self.assertEqual(diffs.names1, ["weight", "bias"])
        self.assertEqual(diffs.names2, ["weight", "bias"])
        self.assertEqual(len(diffs.diffs), 2)
        self.assertIsInstance(diffs.diffs[0], Tensor)
        self.assertIsInstance(diffs.diffs[1], Tensor)

    def test_sorted_param_diffs_with_different_lengths(self) -> None:
        m1 = nn.Linear(10, 2)
        m2 = nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 2))

        with self.assertRaises(ValueError):
            sorted_param_diffs(m1, m2)

    def test_get_gradient(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x**2).sum()
        result = get_gradient(y, x)
        expected = 2 * x
        torch.testing.assert_close(result, expected)

    def test_set_requires_grad(self) -> None:
        m = nn.Linear(10, 2)
        self.assertTrue(m.weight.requires_grad)
        self.assertTrue(m.bias.requires_grad)
        set_requires_grad(m, False)
        self.assertFalse(m.weight.requires_grad)
        self.assertFalse(m.bias.requires_grad)

    def test_to_eval(self) -> None:
        m = nn.Linear(10, 2)
        self.assertTrue(m.training)
        self.assertTrue(m.weight.requires_grad)
        self.assertTrue(m.bias.requires_grad)
        to_eval(m)
        self.assertFalse(m.training)
        self.assertFalse(m.weight.requires_grad)
        self.assertFalse(m.bias.requires_grad)

    def test_toggle_optimizer(self) -> None:
        module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        params = tuple(module.parameters())
        original = (False, True, True, False)
        for param, requires_grad in zip(params, original):
            param.requires_grad_(requires_grad)
        get_requires_grad = lambda: tuple(p.requires_grad for p in params)

        optimizer = torch.optim.SGD(params, lr=0.1)
        context = toggle_optimizer(module, optimizer)
        inner_optimizer = torch.optim.SGD(params[1:3], lr=0.1)
        with patch.object(
            module,
            "parameters",
            wraps=module.parameters,
        ) as parameters, patch.object(
            module,
            "named_parameters",
            wraps=module.named_parameters,
        ) as named_parameters:
            with context:
                self.assertEqual(get_requires_grad(), (True, True, True, True))
        self.assertEqual(get_requires_grad(), original)
        parameters.assert_called_once_with()
        named_parameters.assert_called_once_with(recurse=True)

        optimizer.param_groups[0]["params"] = [params[0], params[3]]
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with context:
                self.assertEqual(get_requires_grad(), (True, False, False, True))
                with toggle_optimizer(module, inner_optimizer):
                    self.assertEqual(
                        get_requires_grad(),
                        (False, True, True, False),
                    )
                self.assertEqual(get_requires_grad(), (True, False, False, True))
                raise RuntimeError("body failed")
        self.assertEqual(get_requires_grad(), original)

    def test_toggle_optimizer_disabled(self) -> None:
        module = Mock()
        with toggle_optimizer(module, object(), enabled=False):
            pass
        self.assertEqual(module.mock_calls, [])

    def test_mode_context(self) -> None:
        def get_modes(module):
            return [m.training for m in module.modules()]

        module = nn.Sequential(
            nn.Sequential(nn.Linear(2, 2), nn.ReLU()),
            nn.Sequential(nn.Dropout(), nn.Linear(2, 2)),
        )
        context = eval_context(module)
        module.train()
        with context as entered:
            self.assertIsNone(entered)
            self.assertTrue(all(not mode for mode in get_modes(module)))
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertTrue(all(get_modes(module)))
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

        module.eval()
        body_error = RuntimeError("body failed")
        with self.assertRaises(RuntimeError) as context_error:
            with context:
                self.assertTrue(all(not mode for mode in get_modes(module)))
                raise body_error
        self.assertIs(context_error.exception, body_error)
        self.assertTrue(all(not mode for mode in get_modes(module)))
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

        with train_context(module) as entered:
            self.assertIsNone(entered)
            self.assertTrue(all(get_modes(module)))
            self.assertTrue(torch.is_grad_enabled())
            with eval_context(module, use_inference=False):
                self.assertTrue(all(not mode for mode in get_modes(module)))
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(torch.is_inference_mode_enabled())
            self.assertTrue(all(get_modes(module)))
            self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(all(not mode for mode in get_modes(module)))

    def test_mode_context_thread_states(self) -> None:
        linear = nn.Linear(3, 1)
        with eval_context(linear):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        with eval_context(linear, use_inference=False):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference_mode_enabled())
        with torch.no_grad():
            with train_context(linear):
                self.assertTrue(torch.is_grad_enabled())
                self.assertFalse(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference_mode_enabled())
        with eval_context(linear, use_grad=True, use_inference=True):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        with mode_context(
            linear,
            to_train=None,
            use_grad=None,
            use_inference=False,
        ):
            linear.eval()
            self.assertTrue(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference_mode_enabled())
        self.assertFalse(linear.training)
        with torch.inference_mode():
            with mode_context(
                linear,
                to_train=None,
                use_grad=True,
                use_inference=False,
            ):
                linear.train()
                self.assertTrue(torch.is_grad_enabled())
                self.assertTrue(torch.is_inference_mode_enabled())
            self.assertTrue(linear.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_mode_context_activates_both_contexts(self) -> None:
        events = []

        class TrackingContext:
            def __init__(self, name, error=None) -> None:
                self.name = name
                self.error = error

            def __enter__(self) -> None:
                events.append(f"{self.name}.enter")

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                events.append(f"{self.name}.exit")
                if self.error is not None:
                    raise self.error

        grad_factory = Mock(side_effect=lambda: TrackingContext("grad"))
        inference_factory = Mock(side_effect=lambda: TrackingContext("inference"))
        context = eval_context(nn.Linear(2, 2))
        with patch("core.learn.toolkit.torch.no_grad", grad_factory):
            with patch(
                "core.learn.toolkit.torch.inference_mode",
                inference_factory,
            ):
                with context:
                    events.append("body")
                with context:
                    events.append("body")

        expected = [
            "grad.enter",
            "inference.enter",
            "body",
            "inference.exit",
            "grad.exit",
        ]
        self.assertListEqual(events, 2 * expected)
        self.assertEqual(grad_factory.call_count, 2)
        self.assertEqual(inference_factory.call_count, 2)

        events.clear()
        inference_error = RuntimeError("inference context exit failed")
        grad_factory = Mock(side_effect=lambda: TrackingContext("grad"))
        inference_factory = Mock(
            side_effect=lambda: TrackingContext("inference", inference_error)
        )
        module = nn.Linear(2, 2)
        with patch("core.learn.toolkit.torch.no_grad", grad_factory):
            with patch(
                "core.learn.toolkit.torch.inference_mode",
                inference_factory,
            ):
                with self.assertRaises(RuntimeError) as context_error:
                    with eval_context(module):
                        events.append("body")
        self.assertIs(context_error.exception, inference_error)
        self.assertListEqual(events, expected)

    def test_mode_context_enter_failure(self) -> None:
        entry_error = RuntimeError("mode enter failed")
        restore_error = RuntimeError("mode restore failed")

        class FailingModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = nn.Linear(2, 2)
                self.errors = {}

            def train(self, mode=True):
                super().train(mode)
                error = self.errors.pop(mode, None)
                if error is not None:
                    raise error
                return self

        module = FailingModule()
        module.errors = {False: entry_error, True: restore_error}
        with self.assertRaises(RuntimeError) as context_error:
            with eval_context(module):
                pass
        self.assertIs(context_error.exception, entry_error)
        self.assertTrue(module.training)
        self.assertTrue(module.child.training)
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

        class EnterFailure:
            def __init__(self, error) -> None:
                self.error = error
                self.events = []

            def __enter__(self) -> None:
                self.events.append("enter")
                raise self.error

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                self.events.append("exit")

        for path, use_inference in [
            ("core.learn.toolkit.torch.no_grad", False),
            ("core.learn.toolkit.torch.inference_mode", True),
        ]:
            with self.subTest(path=path):
                native_error = RuntimeError(f"{path} failed")
                native_context = EnterFailure(native_error)
                module.train()
                with patch(path, return_value=native_context):
                    with self.assertRaises(RuntimeError) as context_error:
                        with eval_context(
                            module,
                            use_inference=use_inference,
                        ):
                            pass
                self.assertIs(context_error.exception, native_error)
                self.assertListEqual(native_context.events, ["enter"])
                self.assertTrue(module.training)
                self.assertTrue(module.child.training)
                self.assertTrue(torch.is_grad_enabled())
                self.assertFalse(torch.is_inference_mode_enabled())

    def test_mode_context_cleanup_failure(self) -> None:
        class FailingRestore(nn.Module):
            def __init__(self, error) -> None:
                super().__init__()
                self.error = error
                self.fail = False
                self.observations = []

            def train(self, mode=True):
                super().train(mode)
                if mode:
                    self.observations.append(
                        (
                            torch.is_grad_enabled(),
                            torch.is_inference_mode_enabled(),
                        )
                    )
                    if self.fail:
                        self.fail = False
                        raise self.error
                return self

        restore_error = RuntimeError("mode restore failed")
        module = FailingRestore(restore_error)
        module.fail = True
        with self.assertRaises(RuntimeError) as context_error:
            with eval_context(module):
                self.assertTrue(all(not m.training for m in module.modules()))
        self.assertIs(context_error.exception, restore_error)
        self.assertTrue(all(m.training for m in module.modules()))
        self.assertEqual(module.observations[-1], (False, True))
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

        module.error = RuntimeError("another mode restore failed")
        module.fail = True
        body_error = RuntimeError("body failed")
        with self.assertRaises(RuntimeError) as context_error:
            with eval_context(module):
                raise body_error
        self.assertIs(context_error.exception, body_error)
        self.assertTrue(all(m.training for m in module.modules()))
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

        class ExitFailure:
            def __init__(self, error) -> None:
                self.error = error

            def __enter__(self) -> None:
                pass

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                raise self.error

        context_exit_error = RuntimeError("context exit failed")
        with patch(
            "core.learn.toolkit.torch.no_grad",
            return_value=ExitFailure(context_exit_error),
        ):
            with self.assertRaises(RuntimeError) as context_error:
                with eval_context(module, use_inference=False):
                    pass
        self.assertIs(context_error.exception, context_exit_error)
        self.assertTrue(all(m.training for m in module.modules()))

        body_error = RuntimeError("body wins")
        with patch(
            "core.learn.toolkit.torch.no_grad",
            return_value=ExitFailure(context_exit_error),
        ):
            with self.assertRaises(RuntimeError) as context_error:
                with eval_context(module, use_inference=False):
                    raise body_error
        self.assertIs(context_error.exception, body_error)
        self.assertTrue(all(m.training for m in module.modules()))
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_eval_context(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        linear = nn.Linear(3, 1)
        y = linear(x)
        with eval_context(linear, use_inference=False):
            z = linear(x)
        self.assertTrue(x.requires_grad)
        self.assertTrue(y.requires_grad)
        self.assertFalse(z.requires_grad)

    def test_no_grad_context(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        linear = nn.Linear(3, 1)
        y = linear(x)
        with no_grad_context(enabled=True):
            z = linear(x)
        self.assertTrue(x.requires_grad)
        self.assertTrue(y.requires_grad)
        self.assertFalse(z.requires_grad)


class TestCheckIsCI(unittest.TestCase):
    def test_check_is_ci_default(self) -> None:
        with patch("sys.argv", ["test_check_is_ci_default"]):
            self.assertFalse(check_is_ci())

    def test_check_is_ci_zero(self) -> None:
        with patch("sys.argv", ["test_check_is_ci_zero", "--ci", "0"]):
            self.assertFalse(check_is_ci())

    def test_check_is_ci_one(self) -> None:
        with patch("sys.argv", ["test_check_is_ci_one", "--ci", "1"]):
            self.assertTrue(check_is_ci())


class TestInitializations(unittest.TestCase):
    def test_initialize_builtin_method(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight
        initializer.initialize(param, "xavier_uniform")
        self.assertIsInstance(param, Tensor)

    def test_initialize_custom_method(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        @Initializer.register("custom")
        def custom(self: Initializer, param: nn.Parameter) -> None:
            with torch.no_grad():
                param.data.fill_(1.0)

        with self.assertRaises(ValueError):

            @Initializer.register("custom")
            def custom(self: Initializer, param: nn.Parameter) -> None:
                pass

        initializer.initialize(param, "custom")
        self.assertIsInstance(param, Tensor)
        torch.testing.assert_close(param, torch.ones_like(param))

    def test_xavier_uniform(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight
        initializer.xavier_uniform(param)
        self.assertIsInstance(param, Tensor)

    def test_xavier_normal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight
        initializer.xavier_normal(param)
        self.assertIsInstance(param, Tensor)

    def test_normal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight
        initializer.normal(param)
        self.assertIsInstance(param, Tensor)

    def test_truncated_normal(self) -> None:
        initializer = Initializer({"span": 0.1, "epoch": 1})
        m = nn.Linear(10, 2)
        param = m.weight
        initializer.truncated_normal(param)
        self.assertIsInstance(param, Tensor)
        initializer = Initializer({"span": 0.1, "epoch": 1, "tol": 1.0})
        initializer.truncated_normal(param)

    def test_orthogonal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight
        initializer.orthogonal(param)
        self.assertIsInstance(param, Tensor)


class TestWeightsStrategy(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def _test_decay(self, num: int, decay: str, expected: np.ndarray) -> None:
        ws = WeightsStrategy(decay)
        weights = ws(num)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(weights.shape, (num,))  # type: ignore
        np.testing.assert_allclose(weights, expected)  # type: ignore

    def test_linear_decay(self) -> None:
        expected_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self._test_decay(10, "linear_decay", expected_weights)

    def test_radius_decay(self) -> None:
        num = 10
        expected_weights = np.sin(np.arccos(1.0 - np.linspace(0, 1, num + 1)[1:]))
        self._test_decay(num, "radius_decay", expected_weights)

    def test_log_decay(self) -> None:
        num = 10
        excepted_weights = np.log(np.arange(num) + np.e)
        self._test_decay(num, "log_decay", excepted_weights)

    def test_sigmoid_decay(self) -> None:
        num = 10
        expected_weights = 1.0 / (1.0 + np.exp(-np.linspace(-5.0, 5.0, num)))
        self._test_decay(num, "sigmoid_decay", expected_weights)

    def test_no_decay(self) -> None:
        ws = WeightsStrategy(None)
        num = 10
        weights = ws(num)
        self.assertIsNone(weights)

    def test_decay_visualize(self) -> None:
        with self.assertRaises(RuntimeError):
            WeightsStrategy(None).visualize(str(self.tmp_path / "no_strategy.png"))
        ws = WeightsStrategy("linear_decay")
        export_path = self.tmp_path / "weights_strategy.png"
        try:
            import matplotlib.pyplot

            ws.visualize(str(export_path))
            self.assertTrue(export_path.is_file())
        except:
            with self.assertRaises(RuntimeError):
                ws.visualize(str(export_path))


class TestWarnOnce(unittest.TestCase):
    def test_warn_once_without_key(self) -> None:
        message = random_hash()
        with patch("core.learn.toolkit.console.warn") as mock_warn:
            warn_once(message)
        mock_warn.assert_called_once_with(message)

    def test_warn_once_with_key(self) -> None:
        key = random_hash()
        message = random_hash()
        with patch("core.learn.toolkit.console.warn") as mock_warn:
            warn_once(message, key=key)
        mock_warn.assert_called_once_with(message)

    def test_warn_once_twice_without_key(self) -> None:
        message = random_hash()
        with patch("core.learn.toolkit.console.warn") as mock_warn:
            warn_once(message)
            warn_once(message)
        mock_warn.assert_called_once_with(message)

    def test_warn_once_twice_with_key(self) -> None:
        key = random_hash()
        message = random_hash()
        with patch("core.learn.toolkit.console.warn") as mock_warn:
            warn_once(message, key=key)
            warn_once(message * 2, key=key)
        mock_warn.assert_called_once_with(message)


class TestToggleModule(unittest.TestCase):
    def setUp(self) -> None:
        self.module = nn.Linear(10, 2)

    def test_toggle_module_enabled(self) -> None:
        with toggle_module(self.module, requires_grad=False):
            for param in self.module.parameters():
                self.assertFalse(param.requires_grad)
        for param in self.module.parameters():
            self.assertTrue(param.requires_grad)

    def test_toggle_module_disabled(self) -> None:
        original_requires_grad = {
            k: p.requires_grad for k, p in self.module.named_parameters()
        }
        with toggle_module(self.module, requires_grad=False, enabled=False):
            for k, param in self.module.named_parameters():
                self.assertEqual(param.requires_grad, original_requires_grad[k])

    def test_toggle_module_restore_original_state(self) -> None:
        original_requires_grad = {
            k: p.requires_grad for k, p in self.module.named_parameters()
        }
        with toggle_module(self.module, requires_grad=False):
            pass
        for k, param in self.module.named_parameters():
            self.assertEqual(param.requires_grad, original_requires_grad[k])


if __name__ == "__main__":
    unittest.main()
