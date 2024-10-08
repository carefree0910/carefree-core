import json
import asyncio
import tempfile
import unittest

import numpy as np

from typing import Any
from typing import Dict
from unittest import mock
from dataclasses import field
from unittest.mock import patch
from unittest.mock import Mock
from core.toolkit.misc import *


test_dict = {}


class TestMisc(unittest.TestCase):
    @patch("core.toolkit.misc.get_ddp_info")
    def test_ddp_info_is_none(self, mock_get_ddp_info):
        mock_get_ddp_info.return_value = None
        self.assertTrue(is_rank_0())

    @patch("core.toolkit.misc.get_ddp_info")
    def test_ddp_info_rank_is_zero(self, mock_get_ddp_info):
        mock_ddp_info = Mock()
        mock_ddp_info.rank = 0
        mock_get_ddp_info.return_value = mock_ddp_info
        self.assertTrue(is_rank_0())

    @patch("core.toolkit.misc.get_ddp_info")
    def test_ddp_info_rank_is_not_zero(self, mock_get_ddp_info):
        mock_ddp_info = Mock()
        mock_ddp_info.rank = 1
        mock_get_ddp_info.return_value = mock_ddp_info
        self.assertFalse(is_rank_0())

    @patch("core.toolkit.misc.is_dist_initialized")
    def test_is_fsdp(self, mock_is_dist_initialized):
        mock_is_dist_initialized.return_value = True
        self.assertFalse(is_fsdp())

    @patch("core.toolkit.misc.is_dist_initialized")
    def test_wait_for_everyone(self, mock_is_dist_initialized):
        mock_is_dist_initialized.return_value = True
        wait_for_everyone()

    def test_walk(self):
        def reset():
            nonlocal paths, hierarchies
            paths = []
            hierarchies = []

        def callback(hierarchy, path):
            paths.append(path)
            hierarchies.append(hierarchy)

        paths = []
        hierarchies = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            (tmp_dir / "a.txt").touch()
            (tmp_dir / "a.png").touch()
            reset()
            walk(str(tmp_dir), callback, {".txt"})
            self.assertListEqual(paths, [str(tmp_dir / "a.txt")])
            self.assertListEqual(hierarchies[0][-2:], [str(tmp_dir.stem), "a.txt"])
            reset()
            walk(str(tmp_dir), callback)
            self.assertSetEqual(
                set(paths), {str(tmp_dir / "a.txt"), str(tmp_dir / "a.png")}
            )

    def test_parse_config(self):
        d = dict(a=0, b=1, c=2)
        self.assertDictEqual(parse_config(None), {})
        self.assertDictEqual(parse_config(d), d)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            config_path = tmp_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(d, f)
            self.assertDictEqual(parse_config(config_path), d)
            self.assertDictEqual(parse_config(str(config_path)), d)

    def test_check_requires(self):
        # test function
        def fn_with_param(a, b, c):
            pass

        self.assertTrue(check_requires(fn_with_param, "a"))
        self.assertTrue(check_requires(fn_with_param, "b"))
        self.assertTrue(check_requires(fn_with_param, "c"))
        self.assertFalse(check_requires(fn_with_param, "d"))

        # test function with variable keyword arguments
        def fn_with_kwargs(a, b, **kwargs):
            pass

        self.assertTrue(check_requires(fn_with_kwargs, "c", strict=False))
        self.assertTrue(check_requires(fn_with_kwargs, "d", strict=False))
        self.assertFalse(check_requires(fn_with_kwargs, "c"))
        self.assertFalse(check_requires(fn_with_kwargs, "d"))

        # test function with variable positional arguments
        def fn_with_varargs(a, b, *args):
            pass

        self.assertFalse(check_requires(fn_with_varargs, "c"))
        self.assertFalse(check_requires(fn_with_varargs, "args"))

        # test class
        class ClassWithParam:
            def __init__(self, a, b):
                pass

        self.assertTrue(check_requires(ClassWithParam, "a"))
        self.assertTrue(check_requires(ClassWithParam, "b"))
        self.assertFalse(check_requires(ClassWithParam, "c"))

    def test_get_requirements(self):
        # test function with parameters
        def fn_with_params(a, b, c):
            pass

        self.assertListEqual(get_requirements(fn_with_params), ["a", "b", "c"])

        # test function without parameters
        def fn_without_params():
            pass

        self.assertListEqual(get_requirements(fn_without_params), [])

        # test function with variable keyword arguments
        def fn_with_kwargs(a, b, **kwargs):
            pass

        self.assertListEqual(get_requirements(fn_with_kwargs), ["a", "b"])

        # test function with default parameters
        def fn_with_defaults(a, b, c=1):
            pass

        self.assertListEqual(get_requirements(fn_with_defaults), ["a", "b"])

        # test function with variable positional arguments
        def fn_with_varargs(a, b, *args):
            pass

        self.assertListEqual(get_requirements(fn_with_varargs), ["a", "b"])

        # test class with parameters in __init__ method
        class ClassWithParams:
            def __init__(self, a, b):
                pass

        self.assertListEqual(get_requirements(ClassWithParams), ["a", "b"])

        # test class without parameters in __init__ method
        class ClassWithoutParams:
            def __init__(self):
                pass

        self.assertListEqual(get_requirements(ClassWithoutParams), [])

    def test_filter_kw(self):
        def fn(a, b):
            pass

        self.assertDictEqual(filter_kw(fn, dict(a=1, b=2, c=3)), dict(a=1, b=2))

    def test_safe_execute(self):
        def fn(a, b):
            pass

        d = dict(a=1, b=2, c=3)
        safe_execute(fn, d)
        with self.assertRaises(TypeError):
            fn(**d)

    def test_safe_instantiate(self):
        class Class:
            def __init__(self, a, b) -> None:
                pass

        d = dict(a=1, b=2, c=3)
        safe_instantiate(Class, d)
        with self.assertRaises(TypeError):
            Class(**d)

    def test_get_num_positional_args(self):
        # test function with positional only parameters
        def fn_pos_only(a, b, /):
            pass

        self.assertEqual(get_num_positional_args(fn_pos_only), 2)

        # test function with positional or keyword parameters
        def fn_pos_or_kw(a, b):
            pass

        self.assertEqual(get_num_positional_args(fn_pos_or_kw), 2)

        # test function with variable positional arguments
        def fn_var_pos(*args):
            pass

        self.assertEqual(get_num_positional_args(fn_var_pos), math.inf)

        # test function with both positional only and positional or keyword parameters
        def fn_mixed(a, /, b):
            pass

        self.assertEqual(get_num_positional_args(fn_mixed), 2)

        # test function with no parameters
        def fn_no_params():
            pass

        self.assertEqual(get_num_positional_args(fn_no_params), 0)

    def test_prepare_workspace_from(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test when workspace directory exists with old directories
            old_dir = os.path.join(
                tmp_dir, (datetime.now() - timedelta(31)).strftime(TIME_FORMAT)
            )
            os.makedirs(old_dir)
            new_workspace = prepare_workspace_from(tmp_dir, timeout=timedelta(30))
            self.assertFalse(os.path.exists(old_dir))
            self.assertTrue(os.path.exists(new_workspace))

            # test when workspace directory doesn't exist
            non_existent_dir = os.path.join(tmp_dir, "non_existent")
            new_workspace = prepare_workspace_from(non_existent_dir)
            self.assertTrue(os.path.exists(new_workspace))

            # test when make parameter is set to False
            non_existent_dir = os.path.join(tmp_dir, "non_existent_2")
            new_workspace = prepare_workspace_from(non_existent_dir, make=False)
            self.assertFalse(os.path.exists(new_workspace))

    def test_get_latest_workspace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test when root directory doesn't exist
            non_existent_dir = os.path.join(tmp_dir, "non_existent")
            self.assertIsNone(get_latest_workspace(non_existent_dir))

            # test when root directory exists but doesn't contain any directories
            self.assertIsNone(get_latest_workspace(tmp_dir))

            # test when root directory contains directories with names that can be parsed as dates
            date_dir_1 = os.path.join(
                tmp_dir, (datetime.now() - timedelta(1)).strftime(TIME_FORMAT)
            )
            date_dir_2 = os.path.join(tmp_dir, datetime.now().strftime(TIME_FORMAT))
            os.makedirs(date_dir_1)
            os.makedirs(date_dir_2)
            self.assertEqual(str(get_latest_workspace(tmp_dir)), date_dir_2)

            # test when root directory contains directories with names that cannot be parsed as dates
            non_date_dir = os.path.join(tmp_dir, "non_date")
            os.makedirs(non_date_dir)
            self.assertEqual(str(get_latest_workspace(tmp_dir)), date_dir_2)

    def test_sort_dict_by_value(self) -> None:
        d = {"a": 2.0, "b": 1.0, "c": 3.0}
        self.assertSequenceEqual(list(sort_dict_by_value(d)), ["b", "a", "c"])

    def test_parse_args(self):
        space = Namespace(a=1, b=2, c=3)
        self.assertEqual(parse_args(space), space)

    def test_get_arguments(self) -> None:
        def _1(a: int = 1, b: int = 2, c: int = 3) -> Dict[str, Any]:
            return get_arguments()

        class _2:
            def __init__(self, a: int = 1, b: int = 2, c: int = 3):
                self.kw = get_arguments()

        self.assertDictEqual(_1(), dict(a=1, b=2, c=3))
        self.assertDictEqual(_2().kw, dict(a=1, b=2, c=3))

    def test_get_arguments_default(self):
        def dummy_function(a, b, c):
            return get_arguments()

        arguments = dummy_function(1, 2, 3)
        expected_arguments = {"a": 1, "b": 2, "c": 3}
        self.assertDictEqual(arguments, expected_arguments)

    def test_get_arguments_num_back(self):
        def dummy_function_1(a, b, c, /, d, *, e):
            return dummy_function_2()

        def dummy_function_2():
            return get_arguments(num_back=1)

        arguments = dummy_function_1(1, 2, 3, 4, e=5)
        expected_arguments = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        self.assertDictEqual(arguments, expected_arguments)

    def test_get_arguments_pop_class_attributes(self):
        class DummyClass:
            def __init__(self, a, b, c):
                self.args = get_arguments(pop_class_attributes=False)

        dummy_object = DummyClass(1, 2, 3)
        expected_arguments = {"self": dummy_object, "a": 1, "b": 2, "c": 3}
        self.assertDictEqual(dummy_object.args, expected_arguments)

    @patch("core.toolkit.misc.inspect.currentframe")
    def test_get_arguments_no_frame(self, mock_currentframe):
        mock_currentframe.return_value = None
        with self.assertRaises(ValueError):
            get_arguments()
        f_mock = Mock()
        mock_currentframe.return_value = f_mock
        f_mock.f_back = None
        with self.assertRaises(ValueError):
            get_arguments()

    def test_get_arguments_frame_backword_fail(self):
        with self.assertRaises(ValueError):
            get_arguments(num_back=100)

    def test_timestamp(self):
        for i in range(2):
            try:
                t0 = timestamp(simplify=True)
                t1 = timestamp()
                time.sleep(0.1)
                t2 = timestamp(simplify=True)
                t3 = timestamp()
                self.assertEqual(t0, t2)
                self.assertEqual(t1, t3)
                break
            except:
                if i == 1:
                    raise
        t0 = timestamp(ensure_different=True)
        t1 = timestamp(ensure_different=True)
        self.assertNotEqual(t0, t1)

    def test_prod(self):
        numbers = range(1, 6)
        self.assertEqual(prod(numbers), 120)

    def test_hash_code(self):
        random_str1, random_str2 = str(random.random()), str(random.random())
        hash11, hash21 = map(hash_code, [random_str1, random_str2])
        hash12, hash22 = map(hash_code, [random_str1, random_str2])
        self.assertEqual(hash11, hash12)
        self.assertEqual(hash21, hash22)
        self.assertNotEqual(hash11, hash22)

    def test_hash_dict(self):
        d0 = dict(a=1, b=2, c=dict(d={3}))
        d1 = dict(c=dict(d={3}), b=2, a=1)
        d2 = dict(a=1, b=2, c=4)
        d3 = dict(a=1, b=2, d=dict(e={3}))
        self.assertEqual(hash_dict(d0), hash_dict(d1))
        self.assertNotEqual(hash_dict(d0), hash_dict(d2))
        self.assertNotEqual(hash_dict(d0), hash_dict(d3))
        self.assertEqual(
            hash_dict(d0, static_keys=True),
            hash_dict(d3, static_keys=True),
        )
        d3 = dict(a="a", b="b", c="c")
        d4 = dict(c="c", b="b", a="a")
        d5 = dict(a="a", b="b", c="d")
        d6 = dict(a="a", b="b", d="d")
        key_order0 = ["a", "b", "c"]
        key_order1 = ["c", "b", "a"]
        self.assertEqual(hash_str_dict(d3), hash_str_dict(d4))
        self.assertEqual(hash_str_dict(d3), hash_str_dict(d3, key_order=key_order0))
        self.assertNotEqual(hash_str_dict(d3), hash_str_dict(d3, key_order=key_order1))
        self.assertNotEqual(hash_str_dict(d3), hash_str_dict(d5))
        self.assertNotEqual(hash_str_dict(d5), hash_str_dict(d6))
        self.assertEqual(
            hash_str_dict(d5, static_keys=True),
            hash_str_dict(d6, static_keys=True),
        )

    def test_prefix_dict(self):
        prefix = "^_^"
        d = {"a": 1, "b": 2, "c": 3}
        self.assertDictEqual(
            prefix_dict(d, prefix),
            {"^_^_a": 1, "^_^_b": 2, "^_^_c": 3},
        )

    def test_shallow_copy_dict(self):
        d = {"a": {"b": 1}}
        dc1 = shallow_copy_dict(d)
        dc1["a"]["b"] = 2
        self.assertEqual(d["a"]["b"], 1)
        dc2 = d.copy()
        dc2["a"]["b"] = 2
        self.assertEqual(d["a"]["b"], 2)
        d = {"a": {"b": [{"c": 1}]}}
        dc1 = shallow_copy_dict(d)
        dc1["a"]["b"][0]["c"] = 2
        self.assertEqual(d["a"]["b"][0]["c"], 1)
        dc2 = d.copy()
        dc2["a"]["b"][0]["c"] = 2
        self.assertEqual(d["a"]["b"][0]["c"], 2)

    def test_update_dict(self):
        src_dict = {"a": 1, "b": {"c": 2}}
        tgt_dict = {"b": {"c": 1, "d": 2}}
        update_dict(src_dict, tgt_dict)
        self.assertDictEqual(tgt_dict, {"a": 1, "b": {"c": 2, "d": 2}})

    def test_fix_float_to_length(self):
        self.assertEqual(fix_float_to_length(1, 8), "1.000000")
        self.assertEqual(fix_float_to_length(1.0, 8), "1.000000")
        self.assertEqual(fix_float_to_length(1.0, 8), "1.000000")
        self.assertEqual(fix_float_to_length(-1, 8), "-1.00000")
        self.assertEqual(fix_float_to_length(-1.0, 8), "-1.00000")
        self.assertEqual(fix_float_to_length(-1.0, 8), "-1.00000")
        self.assertEqual(fix_float_to_length(1234567, 8), "1234567.")
        self.assertEqual(fix_float_to_length(12345678, 8), "12345678")
        self.assertEqual(fix_float_to_length(123456789, 8), "123456789")
        self.assertEqual(fix_float_to_length(1.0e-13, 14), "0.000000000000")
        self.assertEqual(fix_float_to_length(1.0e-12, 14), "0.000000000001")
        self.assertEqual(fix_float_to_length(1.0e-11, 14), "0.000000000010")
        self.assertEqual(fix_float_to_length(-1.0e-12, 14), "-0.00000000000")
        self.assertEqual(fix_float_to_length(-1.0e-11, 14), "-0.00000000001")
        self.assertEqual(fix_float_to_length(-1.0e-10, 14), "-0.00000000010")
        self.assertEqual("+" + fix_float_to_length(math.nan, 8) + "+", "+  nan   +")

    def test_truncate_string_to_length(self):
        self.assertEqual(truncate_string_to_length("123456", 6), "123456")
        self.assertEqual(truncate_string_to_length("1234567", 6), "12..67")
        self.assertEqual(truncate_string_to_length("12345678", 6), "12..78")
        self.assertEqual(truncate_string_to_length("12345678", 7), "12...78")

    def test_grouped(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(grouped(lst, 3), [(1, 2, 3), (4, 5, 6)])
        self.assertEqual(
            grouped(lst, 3, keep_tail=True),
            [(1, 2, 3), (4, 5, 6), (7, 8)],
        )
        unit = 10**5
        lst = list(range(3 * unit + 1))
        gt = [
            tuple(range(unit)),
            tuple(range(unit, 2 * unit)),
            tuple(range(2 * unit, 3 * unit)),
            (3 * unit,),
        ]
        self.assertEqual(grouped(lst, unit, keep_tail=True), gt)
        self.assertEqual(grouped(lst, unit), gt[:-1])

    def test_grouped_into(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(grouped_into(lst, 3), [(1, 2, 3), (4, 5, 6), (7, 8)])

    def test_is_numeric(self):
        self.assertEqual(is_numeric(0x1), True)
        self.assertEqual(is_numeric(1e0), True)
        self.assertEqual(is_numeric("1"), True)
        self.assertEqual(is_numeric("1."), True)
        self.assertEqual(is_numeric("1.0"), True)
        self.assertEqual(is_numeric("1.00"), True)
        self.assertEqual(is_numeric("1.0.0"), False)
        self.assertEqual(is_numeric("Ⅱ"), True)
        self.assertEqual(is_numeric("nan"), True)

    def test_register_core(self):
        global test_dict

        def before_register(cls):
            cls.before = "test_before"

        def after_register(cls):
            self.assertEqual(cls.before, "test_before")
            cls.after = "test_after"

        def register(name):
            return register_core(
                name,
                test_dict,
                before_register=before_register,
                after_register=after_register,
            )

        @register("foo")
        class Foo:
            pass

        self.assertIs(Foo, test_dict["foo"])
        self.assertEqual(Foo.after, "test_after")

        @register("foo")
        class Foo2:
            pass

        self.assertIs(Foo, test_dict["foo"])
        self.assertEqual(Foo2.before, "test_before")
        with self.assertRaises(AttributeError):
            _ = Foo2.after

    def test_get_err_msg(self):
        try:
            raise RuntimeError("test")
        except RuntimeError as err:
            get_err_msg(err)

    def test_offload(self):
        async def sleep():
            time.sleep(unit)

        async def main():
            t0 = time.time()
            await asyncio.gather(*[sleep() for _ in range(num_tasks)])
            t1 = time.time()
            await asyncio.gather(*[offload(sleep()) for _ in range(num_tasks)])
            t2 = time.time()
            self.assertGreater(t1 - t0, unit * (num_tasks - 1))
            self.assertLess(t2 - t1, unit * 0.5 * num_tasks)

        unit = 0.01
        num_tasks = 50
        asyncio.run(main())

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

    def test_to_set_with_set(self):
        inp = {1, 2, 3}
        result = to_set(inp)
        self.assertEqual(result, inp)

    def test_to_set_with_list(self):
        inp = [1, 2, 3]
        result = to_set(inp)
        expected_result = {1, 2, 3}
        self.assertEqual(result, expected_result)

    def test_to_set_with_tuple(self):
        inp = (1, 2, 3)
        result = to_set(inp)
        expected_result = {1, 2, 3}
        self.assertEqual(result, expected_result)

    def test_to_set_with_dict(self):
        inp = {1: "a", 2: "b", 3: "c"}
        result = to_set(inp)
        expected_result = {1, 2, 3}
        self.assertEqual(result, expected_result)

    def test_to_set_with_single_value(self):
        inp = 1
        result = to_set(inp)
        expected_result = {1}
        self.assertEqual(result, expected_result)

    def test_func_decorators(self):
        @only_execute_on_rank0
        @only_execute_on_local_rank0
        def dummy_function(a, b, c):
            pass

        result = dummy_function(1, 2, 3)
        self.assertEqual(result, None)

    def test_get_memory_size_with_int(self):
        result = get_memory_size(1)
        self.assertEqual(result, sys.getsizeof(1))
        result = get_memory_mb(1)
        self.assertAlmostEqual(result, sys.getsizeof(1) / 1024 / 1024)

    def test_get_memory_size_with_str(self):
        result = get_memory_size("test")
        self.assertEqual(result, sys.getsizeof("test"))

    def test_get_memory_size_with_list(self):
        result = get_memory_size([1, 2, 3])
        expected_result = sys.getsizeof([1, 2, 3]) + sys.getsizeof(1) * 3
        self.assertEqual(result, expected_result)
        result = get_memory_size([1, 2, 3, 1])
        expected_result = sys.getsizeof([1, 2, 3, 1]) + sys.getsizeof(1) * 3
        self.assertEqual(result, expected_result)

    def test_get_memory_size_with_dict(self):
        result = get_memory_size({"a": 1, "b": 2, "c": 3})
        expected_result = (
            sys.getsizeof({"a": 1, "b": 2, "c": 3})
            + sys.getsizeof("a") * 3
            + sys.getsizeof(1) * 3
        )
        self.assertEqual(result, expected_result)

    def test_get_memory_size_with_class(self):
        class Foo:
            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c

        foo = Foo(a=1, b=2, c=3)
        d = foo.__dict__
        result = get_memory_size(foo)
        expected_result = (
            sys.getsizeof(foo)
            + sys.getsizeof(d)
            + sys.getsizeof("a") * 3
            + sys.getsizeof(1) * 3
        )
        self.assertEqual(result, expected_result)

    def test_get_memory_size_with_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = get_memory_size(arr)
        self.assertEqual(result, arr.nbytes)

    def test_get_memory_size_with_pandas_index(self):
        import pandas as pd

        idx = pd.Index([1, 2, 3])
        result = get_memory_size(idx)
        self.assertEqual(result, idx.memory_usage(deep=True))

    def test_get_memory_size_with_pandas_dataframe(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_memory_size(df)
        self.assertEqual(result, df.memory_usage(deep=True).sum())

    def test_data_class_base(self):
        @dataclass
        class Foo(DataClassBase):
            bar: int = 0

        foo = Foo()
        foo2 = Foo(1)
        self.assertListEqual(["bar"], [f.name for f in foo.fields])
        self.assertListEqual(["bar"], foo.field_names)
        self.assertListEqual([0], foo.attributes)
        self.assertEqual(foo, foo.copy())
        self.assertEqual(foo, foo.construct(dict(bar=0)))
        foo.update_with(foo2)
        self.assertEqual(foo2, foo)

        @dataclass
        class FooPure:
            bar: int = 0

        @dataclass
        class Bar(DataClassBase):
            a: int = 0
            b: dict = field(default_factory=lambda: {"c": 0})
            d: list = field(default_factory=lambda: [0])
            e: Foo = field(default_factory=Foo)
            f: FooPure = field(default_factory=FooPure)
            g: List[FooPure] = field(default_factory=lambda: [FooPure()])
            h: Dict[str, FooPure] = field(default_factory=lambda: {"i": FooPure()})

        bar = Bar()
        self.assertEqual(bar, bar.copy())

        @dataclass
        class FooExtended(DataClassBase):
            bar: int = 0
            baz: int = 1

        self.assertEqual(Foo().to_hash(), Foo(0).to_hash())
        self.assertEqual(Foo().to_hash(), FooExtended().to_hash())
        self.assertEqual(Foo().to_hash(), FooExtended(baz=1).to_hash())
        self.assertNotEqual(Bar().to_hash(), Bar(1).to_hash())
        self.assertEqual(Bar().to_hash(), Bar(1).to_hash(focuses="b"))
        self.assertNotEqual(Foo().to_hash(), Foo(1).to_hash())
        self.assertEqual(Foo().to_hash(), Foo(1).to_hash(excludes="bar"))

        @dataclass
        class FooComplex(DataClassBase):
            foo: FooExtended = field(default_factory=FooExtended)
            bar: int = 0

        self.assertEqual(FooComplex().asdict(), {"foo": {"bar": 0, "baz": 1}, "bar": 0})
        self.assertEqual(FooComplex().as_modified_dict(), {})
        foo_complex = FooComplex()
        foo_complex.bar = 1
        self.assertEqual(foo_complex.as_modified_dict(), {"bar": 1})
        foo_complex = FooComplex()
        foo_complex.foo.bar = 1
        self.assertEqual(foo_complex.as_modified_dict(), {"foo": {"bar": 1}})
        foo_complex.bar = 2
        self.assertEqual(foo_complex.as_modified_dict(), {"foo": {"bar": 1}, "bar": 2})

        from pydantic.dataclasses import dataclass as pydantic_dataclass

        @dataclass
        class Bar(DataClassBase):
            pass

        @pydantic_dataclass
        class BarPydantic(DataClassBase):
            pass

        with self.assertRaises(TypeError):
            Bar(a=1)
        BarPydantic(a=1)

    def test_with_register(self):
        class Foo(WithRegister):
            d = {}

        @Foo.register("a")
        class A(Foo):
            pass

        @Foo.register("b")
        class B(Foo):
            pass

        self.assertIs(Foo.get("a"), A)
        self.assertTrue(Foo.has("a"))
        self.assertFalse(Foo.has("c"))
        self.assertIsInstance(Foo.make("a", {}), A)
        self.assertIsInstance(Foo.make("a", {}, ensure_safe=True), A)
        self.assertIsInstance(Foo.make_multiple("a"), A)
        made = Foo.make_multiple(["a", "b"])
        self.assertIsInstance(made[0], A)
        self.assertIsInstance(made[1], B)

        @Foo.register("c")
        class C:
            pass

        self.assertTrue(Foo.check_subclass("a"))
        self.assertTrue(Foo.check_subclass("b"))
        self.assertFalse(Foo.check_subclass("c"))

    def test_incrementer(self):
        with self.assertRaises(ValueError):
            Incrementer("3")
        with self.assertRaises(ValueError):
            Incrementer(1)
        Incrementer(2)
        sequence = np.random.random(100)
        incrementer = Incrementer()
        self.assertTrue(incrementer.is_empty)
        self.assertFalse(incrementer.is_full)
        for i, n in enumerate(sequence):
            incrementer.update(n)
            sub_sequence = sequence[: i + 1]
            mean, std = incrementer.mean, incrementer.std
            self.assertTrue(
                np.allclose(
                    [mean, std],
                    [sub_sequence.mean(), sub_sequence.std()],
                )
            )
        window_sizes = [3, 10, 30, 70]
        for window_size in window_sizes:
            incrementer = Incrementer(window_size)
            for i, n in enumerate(sequence):
                incrementer.update(n)
                if i < window_size:
                    sub_sequence = sequence[: i + 1]
                else:
                    sub_sequence = sequence[i - window_size + 1 : i + 1]
                mean, std = incrementer.mean, incrementer.std
                self.assertTrue(
                    np.allclose(
                        [mean, std],
                        [sub_sequence.mean(), sub_sequence.std()],
                    )
                )

    def test_format_float(self):
        self.assertEqual(format_float(1.0), "1.000000")
        self.assertEqual(format_float(1.0e6), "1.0000e+06")


class TestRetry(unittest.TestCase):
    def setUp(self):
        self.counter = 0

    async def async_fn(self):
        self.counter += 1
        if self.counter > 2:
            return "success"
        return "failure"

    def health_check(self, response):
        return response == "success"

    def error_verbose(self, message):
        self.counter = -1

    def test_retry_success_after_retries(self):
        self.counter = 0
        result = asyncio.run(retry(self.async_fn, 3, health_check=self.health_check))
        self.assertEqual(result, "success")
        self.assertEqual(self.counter, 3)

    def test_retry_never_succeeds(self):
        self.counter = 0
        with self.assertRaises(ValueError):
            asyncio.run(retry(self.async_fn, health_check=self.health_check))
        self.assertEqual(self.counter, 1)
        self.counter = 0
        with self.assertRaises(ValueError):
            asyncio.run(retry(self.async_fn, 2, health_check=self.health_check))
        self.assertEqual(self.counter, 2)
        self.counter = 0
        with self.assertRaises(ValueError):
            asyncio.run(
                retry(
                    self.async_fn,
                    2,
                    health_check=self.health_check,
                    error_verbose_fn=self.error_verbose,
                )
            )
        self.assertEqual(self.counter, -1)

    def test_retry_raises_exception(self):
        async def async_fn_raises_exception():
            raise Exception("test exception")

        with self.assertRaises(Exception):
            asyncio.run(retry(async_fn_raises_exception, num_retry=1))


class TestCompress(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_compress_remove_original(self):
        test_dir = os.path.join(self.test_dir, "test_dir0")
        os.makedirs(test_dir)

        # create a file in the test directory
        with open(os.path.join(test_dir, "test_file.txt"), "w") as f:
            f.write("test")

        # compress the test directory and remove the original
        compress(test_dir, remove_original=True)

        # check that the original directory has been removed
        self.assertFalse(os.path.exists(test_dir))

        # check that the zip file exists
        self.assertTrue(os.path.exists(f"{test_dir}.zip"))

    def test_compress_keep_original(self):
        test_dir = os.path.join(self.test_dir, "test_dir1")
        os.makedirs(test_dir)

        # create a file in the test directory
        with open(os.path.join(test_dir, "test_file.txt"), "w") as f:
            f.write("test")

        # compress the test directory and keep the original
        compress(test_dir, remove_original=False)

        # check that the original directory still exists
        self.assertTrue(os.path.exists(test_dir))

        # check that the zip file exists
        self.assertTrue(os.path.exists(f"{test_dir}.zip"))


class TestSerializations(unittest.TestCase):
    def setUp(self):
        @dataclass
        class Foo(ISerializableDataClass["Foo"]):
            key: str = ""

        Foo.d = {}
        Foo.register("foo")(Foo)
        self.Foo = Foo

        class FooArrays(ISerializableArrays["FooArrays"]):
            d = {}

            def __init__(self, key: str = "") -> None:
                self.key = key
                self.array = np.random.randn(3, 5, 7)

            def __eq__(self, other: "FooArrays") -> bool:
                return self.key == other.key and np.allclose(self.array, other.array)

            def to_info(self) -> Dict[str, Any]:
                return dict(key=self.key)

            def from_info(self: "FooArrays", info: Dict[str, Any]) -> "FooArrays":
                self.key = info["key"]
                return self

            def to_npd(self) -> np_dict_type:
                return dict(array=self.array)

            def from_npd(self, npd: np_dict_type) -> "FooArrays":
                self.array = npd["array"]
                return self

        FooArrays.register("foo")(FooArrays)
        self.FooArrays = FooArrays

    def test_serializable(self):
        f = self.Foo("test")
        self.assertEqual(f.to_pack(), JsonPack("foo", dict(key="test")))
        self.assertEqual(f, f.copy())
        self.assertEqual(f, self.Foo.from_pack(f.to_pack().asdict()))
        self.assertEqual(f, self.Foo.from_json(f.to_json()))

        f = self.FooArrays("test")
        self.assertEqual(f.to_pack(), JsonPack("foo", dict(key="test")))
        self.assertEqual(f, f.copy())
        self.assertEqual(
            f, self.FooArrays.from_pack(f.to_pack().asdict()).from_npd(f.to_npd())
        )
        self.assertEqual(f, self.FooArrays.from_json(f.to_json()).from_npd(f.to_npd()))

    def test_serializer(self):
        f = self.Foo("test")
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                Serializer.save_info(tmp_dir)
            Serializer.save_info(tmp_dir, serializable=f)
            self.assertDictEqual(f.to_info(), Serializer.load_info(tmp_dir))
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertIsNone(Serializer.try_load_info(tmp_dir))
            with self.assertRaises(ValueError):
                Serializer.try_load_info(tmp_dir, strict=True)

        f = self.FooArrays("test")
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                Serializer.save_npd(tmp_dir)
            Serializer.save_npd(tmp_dir, serializable=f)
            np.testing.assert_allclose(f.array, Serializer.load_npd(tmp_dir)["array"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                Serializer.load_npd(tmp_dir)
            Serializer.save(tmp_dir, f)
            self.assertEqual(f, Serializer.load(tmp_dir, self.FooArrays))

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                Serializer.load_empty(tmp_dir, self.Foo)


class TestOPTBase(unittest.TestCase):

    def setUp(self):
        from pydantic import Field

        class FooOpt(BaseModel):
            bar: str = "bar"

        class OPTTest(OPTBase):
            foo: FooOpt = Field(default_factory=FooOpt)

            @property
            def env_key(self) -> str:
                return "TEST_ENV_KEY"

        self.opt_factory = lambda: OPTTest()

    def test_init(self):
        self.assertEqual(self.opt_factory().model_dump(), {"foo": {"bar": "bar"}})

    def test_opt_context(self):
        opt = self.opt_factory()
        with opt.opt_context({"foo": {"bar": "updated_bar"}}):
            self.assertEqual(opt.foo.bar, "updated_bar")
        self.assertEqual(opt.foo.bar, "bar")

    def test_opt_env_context(self):
        opt = self.opt_factory()
        with opt.opt_env_context({"foo": {"bar": "env_context_value"}}):
            self.assertEqual(
                json.loads(os.environ[opt.env_key])["foo"]["bar"],
                "env_context_value",
            )
        self.assertNotIn(opt.env_key, os.environ)

    def test_update_from_env(self):
        opt = self.opt_factory()
        with opt.opt_env_context({"foo": {"bar": "updated_bar"}}):
            with opt.opt_env_context({"foo": {"bar": "second_updated_bar"}}):
                opt.update_from_env()
                self.assertEqual(opt.foo.bar, "second_updated_bar")
            opt.update_from_env()
            self.assertEqual(opt.foo.bar, "updated_bar")


class TestTimeit(unittest.TestCase):
    def setUp(self):
        self.message = "test"
        self.precision = 6
        self.timeit = timeit(self.message, precision=self.precision)

    def test_init(self):
        self.assertEqual(self.timeit.message, self.message)
        self.assertEqual(self.timeit.p, self.precision)

    def test_enter(self):
        with mock.patch("time.time", return_value=1234567890.123456):
            self.timeit.__enter__()
            self.assertEqual(self.timeit.t, 1234567890.123456)

    def test_exit(self):
        with mock.patch("time.time", return_value=1234567890.123456):
            self.timeit.__enter__()
        with mock.patch("time.time", return_value=1234567891.123456):
            with mock.patch("core.toolkit.misc.console.log") as mock_log:
                self.timeit.__exit__(None, None, None)
        mock_log.assert_called_once_with(
            f"timing for {self.message:^16s} : {1.000000:6.4f}",
            _stack_offset=3,
        )


class TestBatchManager(unittest.TestCase):
    def setUp(self):
        with self.assertRaises(ValueError):
            batch_manager()
        self.inputs = (np.arange(5), np.arange(5) + 1)
        self.batch_manager = batch_manager(*self.inputs, batch_size=2)
        batch_manager_1 = batch_manager(*self.inputs, num_elem=20)
        self.assertEqual(self.batch_manager.batch_size, batch_manager_1.batch_size)

    def test_init(self):
        self.assertEqual(self.batch_manager.num_samples, 5)
        self.assertEqual(self.batch_manager.batch_size, 2)
        self.assertEqual(self.batch_manager.num_epoch, 3)

    def test_enter(self):
        with self.batch_manager as manager:
            self.assertIs(manager, self.batch_manager)

    def test_iter(self):
        iterator = iter(self.batch_manager)
        self.assertEqual(iterator, self.batch_manager)
        self.assertEqual(self.batch_manager.start, 0)
        self.assertEqual(self.batch_manager.end, 2)

    def test_next(self):
        batches = list(self.batch_manager)
        self.assertEqual(len(batches), 3)
        np.testing.assert_allclose(batches[0][0], np.arange(2))
        np.testing.assert_allclose(batches[0][1], np.arange(2) + 1)
        np.testing.assert_allclose(batches[1][0], np.arange(2, 4))
        np.testing.assert_allclose(batches[1][1], np.arange(2, 4) + 1)
        np.testing.assert_allclose(batches[2][0], np.arange(4, 5))
        np.testing.assert_allclose(batches[2][1], np.arange(4, 5) + 1)
        bm = batch_manager(np.arange(5), batch_size=2)
        batches = list(bm)
        self.assertEqual(len(batches), 3)
        np.testing.assert_allclose(batches[0], np.arange(2))
        np.testing.assert_allclose(batches[1], np.arange(2, 4))
        np.testing.assert_allclose(batches[2], np.arange(4, 5))

    def test_len(self):
        self.assertEqual(len(self.batch_manager), 3)


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


if __name__ == "__main__":
    unittest.main()
