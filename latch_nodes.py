from .utils import PromptUtils


class FloatLatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT",),
                "replace": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
            "optional": {
                "float_input": (
                    "INT",
                    {
                        "forceInput": True,
                        "lazy": True,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    TITLE = "Float Latch"

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_NODE = True

    FUNCTION = "execute"

    CATEGORY = "private/latch"

    def check_lazy_status(
        self,
        value: int,
        replace: bool,
        unique_id,
        prompt,
        float_input: int | None = None,
    ):
        # If float_input is connected, require evaluation based on replace
        if replace and float_input is None:
            pu = PromptUtils(prompt)
            if pu.is_input_connected(unique_id, "float_input"):
                return ["float_input"]
        return []

    def execute(
        self,
        value: int,
        replace: bool,
        unique_id,
        prompt,
        float_input: int | None = None,
    ):
        if float_input is not None and replace:
            value = float_input

        return {"ui": {"value": (value,)}, "result": (value,)}


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
                "prompt": "PROMPT",
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
        prompt,
        int_input: int | None = None,
    ):
        # If int_input is connected, require evaluation based on replace
        if replace and int_input is None:
            pu = PromptUtils(prompt)
            if pu.is_input_connected(unique_id, "int_input"):
                return ["int_input"]
        return []

    def execute(
        self,
        value: int,
        replace: bool,
        unique_id,
        prompt,
        int_input: int | None = None,
    ):
        if int_input is not None and replace:
            value = int_input

        return {"ui": {"value": (value,)}, "result": (value,)}


NODE_CLASS_MAPPINGS = {
    "FloatLatch": FloatLatch,
    "IntegerLatch": IntegerLatch,
    # TextLatch is over at ComfyUI-YALLM-node
}
