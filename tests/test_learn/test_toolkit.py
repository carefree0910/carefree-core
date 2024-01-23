import unittest
import subprocess

from core.learn.toolkit import *
from unittest.mock import patch
from safetensors.torch import save_file
from core.toolkit.misc import random_hash


class TestToolkit(unittest.TestCase):
    def test_new_seed(self) -> None:
        for _ in range(10):
            seed = new_seed()
            self.assertLessEqual(seed, max_seed_value)
            self.assertGreaterEqual(seed, min_seed_value)

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

    def test_check_is_ci(self) -> None:
        is_ci_path = str(Path(__file__).parent / "is_ci_task.py")
        subprocess.run(["python", str(is_ci_path), "--ci", "1"], check=True)
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(["python", str(is_ci_path)], check=True)

    def test_get_torch_device(self) -> None:
        cpu = torch.device("cpu")
        cuda1 = torch.device("cuda:1")
        self.assertTrue(get_torch_device(None) == cpu)
        self.assertTrue(get_torch_device("cpu") == cpu)
        self.assertTrue(get_torch_device(1) == cuda1)
        self.assertTrue(get_torch_device("1") == cuda1)
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

    def test_get_file_info(self) -> None:
        text = "This is a test file."
        test_file = Path("test_file.txt")
        test_file.write_text(text)

        file_info = get_file_info(test_file)

        self.assertIsInstance(file_info, FileInfo)
        self.assertEqual(file_info.sha, hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(file_info.st_size, len(text))

        test_file.unlink()

    def test_check_sha_with_matching_hash(self) -> None:
        path = Path("test_file.txt")
        path.write_text("This is a test file.")
        tgt_sha = hashlib.sha256(path.read_bytes()).hexdigest()

        result = check_sha_with(path, tgt_sha)

        self.assertTrue(result)

        path.unlink()

    def test_check_sha_with_non_matching_hash(self) -> None:
        path = Path("test_file.txt")
        path.write_text("This is a test file.")
        tgt_sha = "0"

        result = check_sha_with(path, tgt_sha)

        self.assertFalse(result)

        path.unlink()

    def test_show_or_return(self) -> None:
        plt.figure()
        self.assertIsInstance(show_or_return(True), np.ndarray)
        plt.close()

    def test_get_tensors_from_safetensors(self) -> None:
        path = Path("example.safetensors")
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        save_file(expected_tensors, path)

        tensors = get_tensors(path)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)
        path.unlink()

    def test_get_tensors_from_pt(self) -> None:
        file_path = "example.pt"
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        torch.save(expected_tensors, file_path)

        tensors = get_tensors(file_path)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)
        os.remove(file_path)

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
        }
        expected_states = {
            "a": torch.tensor([1.0, 2.0, 0.0]),
            "b": torch.tensor([4.0, 5.0, 6.0]),
        }

        new_states = fix_denormal_states(states)

        self.assertIsInstance(new_states, dict)
        torch.testing.assert_close(new_states, expected_states)

    def test_has_batch_norms_with_batch_norm_layers(self) -> None:
        m = nn.Sequential(nn.Linear(10, 2), nn.BatchNorm1d(2))

        result = has_batch_norms(m)

        self.assertTrue(result)

    def test_has_batch_norms_without_batch_norm_layers(self) -> None:
        m = nn.Sequential(nn.Linear(10, 2), nn.ReLU())

        result = has_batch_norms(m)

        self.assertFalse(result)

    def test_inject_parameters_with_identical_modules(self):
        src = nn.Linear(10, 2)
        tgt = nn.Linear(10, 2)
        inject_parameters(src, tgt)
        for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
            torch.testing.assert_close(src_param, tgt_param)

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
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.truncated_normal(param)

        self.assertIsInstance(param, Tensor)

    def test_orthogonal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.orthogonal(param)

        self.assertIsInstance(param, Tensor)


class TestWeightsStrategy(unittest.TestCase):
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
        ws = WeightsStrategy("linear_decay")
        try:
            import matplotlib.pyplot

            ws.visualize()
        except:
            with self.assertRaises(RuntimeError):
                ws.visualize()


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


class TestGetDDPInfo(unittest.TestCase):
    def setUp(self) -> None:
        self.original_environ = os.environ.copy()

    def tearDown(self) -> None:
        os.environ = self.original_environ

    def test_get_ddp_info_with_environment_variables_set(self) -> None:
        os.environ["RANK"] = "1"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "3"
        result = get_ddp_info()
        self.assertIsInstance(result, DDPInfo)
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.world_size, 2)
        self.assertEqual(result.local_rank, 3)

    def test_get_ddp_info_with_no_environment_variables_set(self) -> None:
        if "RANK" in os.environ:
            del os.environ["RANK"]
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]
        if "LOCAL_RANK" in os.environ:
            del os.environ["LOCAL_RANK"]
        result = get_ddp_info()
        self.assertIsNone(result)


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
