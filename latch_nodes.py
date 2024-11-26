# Might be time to make a utility class for this kinda stuff...
# TODO This also exists in SimpleBus
def get_node_info(extra_pnginfo, node_id) -> dict:
    node_id = int(node_id)
    for node_info in extra_pnginfo["workflow"]["nodes"]:
        if node_info["id"] == node_id:
            return node_info
    # WTF is this?
    raise RuntimeError(f"Node missing from workflow: {node_id}")


class IntegerLatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT",),
                "replace": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
            "optional": {
                "int_input": (
                    "INT",
                    {
                        "forceInput": True,
                        "lazy": True,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    TITLE = "Integer Latch"

    RETURN_TYPES = ("INT",)
    OUTPUT_NODE = True

    FUNCTION = "execute"

    CATEGORY = "private/latch"

    def check_lazy_status(
        self,
        value: int,
        replace: bool,
        unique_id,
        extra_pnginfo,
        int_input: int | None = None,
    ):
        # If int_input is connected, require evaluation based on replace
        if replace:
            my_node_info = get_node_info(extra_pnginfo, unique_id)
            if any(
                input["name"] == "int_input" and input["link"] is not None
                for input in my_node_info.get("inputs", [])
            ):
                return ["int_input"]
        return []

    def execute(
        self,
        value: int,
        replace: bool,
        unique_id,
        extra_pnginfo,
        int_input: int | None = None,
    ):
        if int_input is not None and replace:
            value = int_input

        return {"ui": {"value": (value,)}, "result": (value,)}


NODE_CLASS_MAPPINGS = {
    "IntegerLatch": IntegerLatch,
}
