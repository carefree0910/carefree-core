import unittest

from core.toolkit.data_structures import *


class TestDataStructures(unittest.TestCase):
    def test_bundle(self):
        dicts = [dict(a=0), dict(b=1), dict(c=2)]
        for no_mapping in [True, False]:
            bundle = Bundle(no_mapping=no_mapping)

            for i, d in enumerate(dicts):
                bundle.push(Item(str(i), d))
            self.assertEqual(len(bundle), 3)
            for i, item in enumerate(bundle):
                self.assertDictEqual(item.data, dicts[i])
            for i in range(10):
                if i < len(dicts):
                    self.assertIn(str(i), bundle)
                    self.assertDictEqual(bundle.get_index(i).data, dicts[i])
                    self.assertDictEqual(bundle.get(str(i)).data, dicts[i])
                else:
                    self.assertNotIn(str(i), bundle)
                    self.assertIsNone(bundle.get(str(i)))
            self.assertDictEqual(bundle.first.data, dicts[0])
            self.assertDictEqual(bundle.last.data, dicts[-1])
            self.assertFalse(bundle.is_empty)
            for i in range(len(dicts)):
                bundle.remove(str(i))
            self.assertIsNone(bundle.first)
            self.assertIsNone(bundle.last)
            self.assertTrue(bundle.is_empty)

            bundle.push(Item("0", 0))
            with self.assertRaises(ValueError):
                bundle.push(Item("0", 1))

        bundle = Bundle(no_mapping=True)
        self.assertIsNone(bundle.remove("foo"))

    def test_types(self):
        types = Types()
        type_names = ["A", "B", "C"]
        type_bases = [type(name, (), {}) for name in type_names]
        for name, base in zip(type_names, type_bases):
            types[name] = base
        self.assertListEqual(list(types), type_names)
        for name, base in zip(type_names, type_bases):
            self.assertIsInstance(types.make(name), base)
        for i, (name, base) in enumerate(types.items()):
            self.assertEqual(name, type_names[i])
            self.assertIs(base, type_bases[i])
        for i, base in enumerate(types.values()):
            self.assertIs(base, type_bases[i])

    def test_pool(self):
        class Item:
            def __init__(self, key: str):
                self.key = key
                alive.add(self.key)

            def load(self):
                activated.add(self.key)

            def unload(self):
                activated.remove(self.key)

            def collect(self):
                alive.remove(self.key)

        class TestPool(Pool[Item]):
            pass

        alive = set()
        activated = set()
        limit = 3
        pool = TestPool(limit)
        all_keys = ["a", "b", "c", "d", "e", "f"]
        init_fns = [(lambda k: lambda: Item(k))(key) for key in all_keys]
        for i, (key, init_fn) in enumerate(zip(all_keys, init_fns)):
            pool.register(key, init_fn)
            for j in range(min(limit, i + 1)):
                self.assertIn(all_keys[j], alive)
            for j in range(min(limit, i + 1), len(all_keys)):
                self.assertNotIn(all_keys[j], alive)
        for i, key in enumerate(all_keys):
            if i < limit:
                self.assertNotIn(key, activated)
                with pool.use(key):
                    self.assertIn(key, activated)
                self.assertNotIn(key, activated)
            else:
                self.assertNotIn(key, alive)
                self.assertNotIn(key, activated)
                if i % 2 == 0:
                    pool.get(key)
                    self.assertIn(key, alive)
                    self.assertNotIn(key, activated)
                    self.assertNotIn(all_keys[i - limit], alive)
                    with pool.use(key):
                        self.assertIn(key, activated)
                else:
                    with pool.use(key):
                        self.assertIn(key, alive)
                        self.assertIn(key, activated)
                        self.assertNotIn(all_keys[i - limit], alive)
                    self.assertNotIn(all_keys[i - limit], alive)
                self.assertIn(key, alive)
                self.assertNotIn(key, activated)

        with self.assertRaises(ValueError):
            TestPool(0)
        pool = TestPool()
        pool.register("a", init_fns[0])
        self.assertIn("a", pool)
        self.assertIs(pool.activated["a"], pool._fetch("a"))
        with self.assertRaises(ValueError):
            pool.register("a", init_fns[0])
        with self.assertRaises(ValueError):
            pool._fetch("b")
        pool = TestPool(allow_duplicate=True)
        pool.register("a", init_fns[0])
        pool.register("a", init_fns[1])
        self.assertIs(pool._fetch("a").init_fn, init_fns[0])
        self.assertEqual(pool.get("a").key, "a")


if __name__ == "__main__":
    unittest.main()
