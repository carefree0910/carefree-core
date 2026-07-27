import unittest

from core.toolkit.registry import Registry


class Plugin:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class FirstPlugin(Plugin):
    pass


class SecondPlugin(Plugin):
    pass


class CountingPlugin(Plugin):
    num_created = 0

    def __init__(self, value: int = 0) -> None:
        super().__init__(value)
        CountingPlugin.num_created += 1


class Unrelated:
    pass


class TestRegistry(unittest.TestCase):
    def test_registration_order_and_make(self) -> None:
        storage = {}
        registry = Registry(storage, base_type=Plugin)
        self.assertEqual(len(registry), 0)
        self.assertFalse(registry)
        self.assertEqual(registry.aliases, {})

        self.assertIs(registry.register("first", FirstPlugin), FirstPlugin)
        self.assertIs(registry.register("second", SecondPlugin), SecondPlugin)

        self.assertIs(registry.storage, storage)
        self.assertEqual(storage, {"first": FirstPlugin, "second": SecondPlugin})
        self.assertEqual(list(registry), ["first", "second"])
        self.assertEqual(list(registry.keys()), ["first", "second"])
        self.assertEqual(list(registry.values()), [FirstPlugin, SecondPlugin])
        self.assertEqual(
            list(registry.items()),
            [("first", FirstPlugin), ("second", SecondPlugin)],
        )
        self.assertIn("first", registry)
        self.assertNotIn("missing", registry)
        self.assertNotIn(1, registry)
        self.assertTrue(registry.has("first"))
        self.assertFalse(registry.has("missing"))
        self.assertEqual(registry.resolve_name("first"), "first")
        self.assertIs(registry.get("first"), FirstPlugin)

        config = {"value": 7}
        made = registry.make("first", config)
        self.assertIsInstance(made, FirstPlugin)
        self.assertEqual(made.value, 7)
        self.assertEqual(config, {"value": 7})
        self.assertIsInstance(registry.make("second"), SecondPlugin)

        with self.assertRaises(KeyError):
            registry.resolve_name("missing")
        with self.assertRaises(KeyError):
            registry.get("missing")
        with self.assertRaises(KeyError):
            registry.make("missing")

    def test_aliases(self) -> None:
        registry = Registry(base_type=Plugin)
        registry.register("first", FirstPlugin)
        registry.register("second", SecondPlugin)

        self.assertEqual(registry.register_alias("primary", "first"), "first")
        self.assertEqual(registry.register_alias("nested", "primary"), "first")
        self.assertEqual(
            registry.aliases,
            {"primary": "first", "nested": "first"},
        )
        self.assertEqual(registry.resolve_name("nested"), "first")
        self.assertIs(registry.get("primary"), FirstPlugin)
        self.assertIsInstance(registry.make("nested", {"value": 3}), FirstPlugin)
        self.assertIn("nested", registry)
        self.assertTrue(registry.has("primary"))
        self.assertEqual(list(registry), ["first", "second"])

        aliases = registry.aliases
        aliases["external"] = "second"
        self.assertNotIn("external", registry.aliases)

        with self.assertRaises(KeyError):
            registry.register_alias("dangling", "missing")

    def test_duplicate_registration_policies(self) -> None:
        with self.assertRaises(ValueError):
            Registry(base_type=Plugin, duplicate="invalid")

        registry = Registry(base_type=Plugin)
        registry.register("first", FirstPlugin)
        with self.assertRaises(ValueError):
            registry.register("first", SecondPlugin)
        self.assertIs(registry.get("first"), FirstPlugin)

        self.assertIs(
            registry.register("first", SecondPlugin, duplicate="keep"),
            SecondPlugin,
        )
        self.assertIs(registry.get("first"), FirstPlugin)

        self.assertIs(
            registry.register("first", SecondPlugin, duplicate="replace"),
            SecondPlugin,
        )
        self.assertIs(registry.get("first"), SecondPlugin)
        self.assertEqual(list(registry), ["first"])

        keep_registry = Registry(base_type=Plugin, duplicate="keep")
        keep_registry.register("first", FirstPlugin)
        keep_registry.register("first", SecondPlugin)
        self.assertIs(keep_registry.get("first"), FirstPlugin)

        replace_registry = Registry(base_type=Plugin, duplicate="replace")
        replace_registry.register("first", FirstPlugin)
        replace_registry.register("second", SecondPlugin)
        replace_registry.register("first", SecondPlugin)
        self.assertEqual(list(replace_registry), ["first", "second"])
        self.assertIs(replace_registry.get("first"), SecondPlugin)

        with self.assertRaises(ValueError):
            registry.register("first", FirstPlugin, duplicate="invalid")

    def test_alias_collision_policies(self) -> None:
        registry = Registry(base_type=Plugin)
        registry.register("first", FirstPlugin)
        registry.register("second", SecondPlugin)
        registry.register_alias("alias", "first")

        with self.assertRaises(ValueError):
            registry.register_alias("alias", "second")
        self.assertEqual(
            registry.register_alias("alias", "second", duplicate="keep"),
            "first",
        )
        self.assertEqual(registry.resolve_name("alias"), "first")
        self.assertEqual(
            registry.register_alias("alias", "second", duplicate="replace"),
            "second",
        )
        self.assertEqual(registry.resolve_name("alias"), "second")

        with self.assertRaises(ValueError):
            registry.register_alias("first", "second")
        self.assertEqual(
            registry.register_alias("first", "second", duplicate="keep"),
            "first",
        )
        with self.assertRaises(ValueError):
            registry.register_alias("first", "second", duplicate="replace")
        self.assertIs(registry.get("first"), FirstPlugin)

        with self.assertRaises(ValueError):
            registry.register("alias", FirstPlugin)
        self.assertIs(
            registry.register("alias", FirstPlugin, duplicate="keep"),
            FirstPlugin,
        )
        self.assertEqual(registry.resolve_name("alias"), "second")
        self.assertIs(
            registry.register("alias", FirstPlugin, duplicate="replace"),
            FirstPlugin,
        )
        self.assertEqual(registry.resolve_name("alias"), "alias")
        self.assertIs(registry.get("alias"), FirstPlugin)
        self.assertEqual(list(registry), ["first", "second", "alias"])

    def test_invalid_registration(self) -> None:
        with self.assertRaises(TypeError):
            Registry(base_type=Plugin())
        with self.assertRaises(TypeError):
            Registry({1: FirstPlugin}, base_type=Plugin)
        with self.assertRaises(ValueError):
            Registry({"": FirstPlugin}, base_type=Plugin)
        with self.assertRaises(TypeError):
            Registry({"invalid": FirstPlugin()}, base_type=Plugin)
        with self.assertRaises(TypeError):
            Registry({"invalid": Unrelated}, base_type=Plugin)

        registry = Registry(base_type=Plugin)
        for invalid_name in ["", 1]:
            with self.subTest(name=invalid_name):
                with self.assertRaises((TypeError, ValueError)):
                    registry.register(invalid_name, FirstPlugin)

        with self.assertRaises(TypeError):
            registry.register("instance", FirstPlugin())
        with self.assertRaises(TypeError):
            registry.register("unrelated", Unrelated)
        registry.register("first", FirstPlugin)
        with self.assertRaises(TypeError):
            registry.make("first", 1)

        unbounded = Registry()
        unbounded.register("unrelated", Unrelated)
        self.assertIsInstance(unbounded.make("unrelated"), Unrelated)

    def test_make_many(self) -> None:
        registry = Registry(base_type=Plugin)
        registry.register("first", FirstPlugin)
        registry.register("second", SecondPlugin)

        self.assertEqual(registry.make_many([], []), [])
        no_configs = registry.make_many(("first", "second"))
        self.assertEqual(
            [type(instance) for instance in no_configs],
            [FirstPlugin, SecondPlugin],
        )

        flat_config = {"value": 1}
        scalar = registry.make_many("first", flat_config)
        self.assertEqual(len(scalar), 1)
        self.assertIsInstance(scalar[0], FirstPlugin)
        self.assertEqual(scalar[0].value, 1)
        self.assertEqual(flat_config, {"value": 1})

        scalar_sequence = registry.make_many("second", [{"value": 2}])
        self.assertEqual(len(scalar_sequence), 1)
        self.assertIsInstance(scalar_sequence[0], SecondPlugin)
        self.assertEqual(scalar_sequence[0].value, 2)

        positional_configs = [{"value": 3}, {"value": 4}]
        positional = registry.make_many(
            ["first", "second"],
            positional_configs,
        )
        self.assertEqual([instance.value for instance in positional], [3, 4])
        self.assertEqual(
            positional_configs,
            [{"value": 3}, {"value": 4}],
        )

        named_configs = {
            "first": {"value": 5},
            "unused": {"value": 6},
        }
        named = registry.make_many(["first", "second"], named_configs)
        self.assertEqual([instance.value for instance in named], [5, 0])
        self.assertEqual(
            named_configs,
            {"first": {"value": 5}, "unused": {"value": 6}},
        )

    def test_make_many_validates_before_construction(self) -> None:
        CountingPlugin.num_created = 0
        registry = Registry(base_type=Plugin)
        registry.register("first", CountingPlugin)
        registry.register("second", CountingPlugin)

        for configs in [
            [],
            [{"value": 1}],
            [{"value": 1}, {"value": 2}, {"value": 3}],
        ]:
            with self.subTest(configs=configs):
                with self.assertRaises(ValueError):
                    registry.make_many(["first", "second"], configs)
                self.assertEqual(CountingPlugin.num_created, 0)

        with self.assertRaises(ValueError):
            registry.make_many("first", [])
        self.assertEqual(CountingPlugin.num_created, 0)
        with self.assertRaises(ValueError):
            registry.make_many("first", [{}, {}])
        self.assertEqual(CountingPlugin.num_created, 0)

        with self.assertRaises(TypeError):
            registry.make_many(["first", "second"], {"first": 1})
        self.assertEqual(CountingPlugin.num_created, 0)
        with self.assertRaises(TypeError):
            registry.make_many(["first", "second"], {"value": 1})
        self.assertEqual(CountingPlugin.num_created, 0)
        with self.assertRaises(TypeError):
            registry.make_many(["first", "second"], {1: {}})
        self.assertEqual(CountingPlugin.num_created, 0)
        with self.assertRaises(TypeError):
            registry.make_many(["first", "second"], 1)
        self.assertEqual(CountingPlugin.num_created, 0)
        with self.assertRaises(TypeError):
            registry.make_many(1)
        self.assertEqual(CountingPlugin.num_created, 0)
        for configs in ["invalid", b"invalid"]:
            with self.subTest(configs=configs):
                with self.assertRaises(TypeError):
                    registry.make_many(["first", "second"], configs)
                self.assertEqual(CountingPlugin.num_created, 0)
        with self.assertRaises(TypeError):
            registry.make_many(["first", "second"], [{}, 1])
        self.assertEqual(CountingPlugin.num_created, 0)
        with self.assertRaises(TypeError):
            registry.make_many(["first", 1])
        self.assertEqual(CountingPlugin.num_created, 0)

    def test_reset_and_isolated_scope(self) -> None:
        storage = {"first": FirstPlugin}
        registry = Registry(storage, base_type=Plugin)
        registry.register_alias("primary", "first")

        registry.reset()
        self.assertEqual(storage, {})
        self.assertEqual(registry.aliases, {})
        registry.reset()

        registry.register("first", FirstPlugin)
        registry.register_alias("primary", "first")
        with registry.isolated():
            registry.register("first", SecondPlugin, duplicate="replace")
            registry.register("second", SecondPlugin)
            registry.register_alias("temporary", "second")
            self.assertIs(registry.get("first"), SecondPlugin)
            self.assertEqual(list(registry), ["first", "second"])
            self.assertEqual(
                registry.aliases,
                {"primary": "first", "temporary": "second"},
            )
        self.assertEqual(list(registry), ["first"])
        self.assertIs(registry.get("first"), FirstPlugin)
        self.assertEqual(registry.aliases, {"primary": "first"})

        with registry.scope():
            registry.register("second", SecondPlugin)
            with registry.isolated():
                registry.reset()
                registry.register("second", SecondPlugin)
                self.assertEqual(list(registry), ["second"])
            self.assertEqual(list(registry), ["first", "second"])
        self.assertEqual(list(registry), ["first"])

        with registry.scope(reset=True):
            self.assertEqual(list(registry), [])
            self.assertEqual(registry.aliases, {})
            registry.register("second", SecondPlugin)
        self.assertEqual(list(registry), ["first"])

        with self.assertRaisesRegex(RuntimeError, "restore"):
            with registry.isolated():
                registry.reset()
                raise RuntimeError("restore")
        self.assertEqual(list(registry), ["first"])
        self.assertEqual(registry.aliases, {"primary": "first"})


if __name__ == "__main__":
    unittest.main()
