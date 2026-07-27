from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import List
    from typing import Type
    from typing_extensions import assert_type
    from core.toolkit.misc import JsonPack
    from core.toolkit.misc import WithRegister
    from core.toolkit.misc import ISerializable

    class Plugin(WithRegister["Plugin"]):
        d: Dict[str, Type["Plugin"]] = {}

        def __init__(self, value: int = 0) -> None:
            self.value = value

    @Plugin.register("typing.plugin")
    class RegisteredPlugin(Plugin):
        pass

    class Payload(ISerializable["Payload"]):
        d: Dict[str, Type["Payload"]] = {}

        def __init__(self, value: int = 0) -> None:
            self.value = value

        def to_info(self) -> Dict[str, Any]:
            return {"value": self.value}

        def from_info(self, info: Dict[str, Any]) -> "Payload":
            self.value = int(info["value"])
            return self

    @Payload.register("typing.payload")
    class RegisteredPayload(Payload):
        pass

    assert_type(RegisteredPlugin, Type[RegisteredPlugin])
    assert_type(Plugin.d, Dict[str, Type[Plugin]])
    assert_type(Plugin.get("typing.plugin"), Type[Plugin])
    assert_type(Plugin.has("typing.plugin"), bool)
    assert_type(Plugin.check_subclass("typing.plugin"), bool)
    assert_type(Plugin.make("typing.plugin", {"value": 1}), Plugin)
    assert_type(
        RegisteredPlugin.get("typing.plugin"),
        Type[Plugin],
    )
    assert_type(
        RegisteredPlugin.make("typing.plugin", {"value": 1}),
        Plugin,
    )
    assert_type(Plugin.make_multiple("typing.plugin"), Plugin)
    assert_type(
        Plugin.make_multiple(
            "typing.plugin",
            {"value": 1},
            ensure_safe=True,
        ),
        Plugin,
    )
    assert_type(
        Plugin.make_multiple(["typing.plugin"], [{"value": 1}]),
        List[Plugin],
    )
    assert_type(
        Plugin.make_multiple(
            ["typing.plugin"],
            {"typing.plugin": {"value": 1}},
        ),
        List[Plugin],
    )
    assert_type(
        RegisteredPlugin.make_multiple(["typing.plugin"]),
        List[Plugin],
    )

    payload = RegisteredPayload(1)
    assert_type(RegisteredPayload, Type[RegisteredPayload])
    assert_type(Payload.get("typing.payload"), Type[Payload])
    assert_type(Payload.make("typing.payload", {"value": 1}), Payload)
    assert_type(
        RegisteredPayload.make("typing.payload", {"value": 1}),
        Payload,
    )
    assert_type(Payload.from_pack({"type": "typing.payload", "info": {}}), Payload)
    assert_type(Payload.from_json('{"type": "typing.payload", "info": {}}'), Payload)
    assert_type(payload.to_pack(), JsonPack)
    assert_type(payload.copy(), RegisteredPayload)
