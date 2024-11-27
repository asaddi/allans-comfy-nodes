from pprint import pprint

import torch

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder


class MaskBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 31, "step": 1}),
                "sigma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
            }
        }

    TITLE = "Mask Blur"

    RETURN_TYPES = ("MASK",)

    FUNCTION = "mask_blur"

    CATEGORY = "private/mask"

    def mask_blur(self, mask: torch.Tensor, blur_radius: int, sigma: float):
        # We're going to be lazy and just do it all via graph expansion
        graph = GraphBuilder()
        m2i = graph.node("MaskToImage", mask=mask)
        blur = graph.node(
            "ImageBlur", image=m2i.out(0), blur_radius=blur_radius, sigma=sigma
        )
        i2m = graph.node("ImageToMask", image=blur.out(0), channel="red")
        return {
            "result": (i2m.out(0),),
            "expand": graph.finalize(),
        }


class AnyType(str):
    def __ne__(self, other):
        if self == "*":
            # If we're the wildcard, match anything
            return False
        return super().__ne__(other)


# Server-side-only implementation of an "any switch"
class PrivateAnySwitch:
    # Being server-side-only, it means we have a fixed number of inputs
    NUM_INPUTS = 2

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

        for i in range(cls.NUM_INPUTS):
            d["optional"][f"input{i + 1}"] = ("*",)

        return d

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # We've had luck using this to prevent validation errors on our
        # wildcard inputs, so we'll just stick with it.
        # (As opposed to using the str subclass, as done below.)
        return True

    TITLE = "Any Switch 2"

    RETURN_TYPES = (AnyType("*"),)

    FUNCTION = "switch"

    CATEGORY = "private/switch"

    def switch(self, unique_id, **kwargs):
        for i in range(self.NUM_INPUTS):
            input = kwargs.get(f"input{i + 1}")
            if input is not None:
                return (input,)
        return (ExecutionBlocker(f"{self.TITLE} (#{unique_id}): All inputs muted"),)


class PrivateAnySwitch4(PrivateAnySwitch):
    NUM_INPUTS = 4

    TITLE = "Any Switch 4"


class DumpToConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_pprint": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "pprint",
                        "label_off": "raw",
                    },
                ),
                "input": ("*",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    TITLE = "Dump To Console"

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "run"

    CATEGORY = "private/debug"

    def run(self, use_pprint, input, unique_id):
        print(f"DumpToConsole (#{unique_id}) = ", end="")
        if use_pprint:
            pprint(input)
        else:
            print(input)

        return ()


class CLIPDistance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip1": ("CONDITIONING",),
                "clip2": ("CONDITIONING",),
            },
        }

    TITLE = "CLIP Distance"

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "calculate"

    CATEGORY = "private/experimental"

    def calculate(self, clip1, clip2):
        clip1: torch.Tensor = clip1[0][0]
        clip2: torch.Tensor = clip2[0][0]

        print(clip1)
        print(clip2)
        dist = torch.dist(clip1, clip2)
        print(f"CLIP distance = {dist}")

        return ()


NODE_CLASS_MAPPINGS = {
    "MaskBlur": MaskBlur,
    "AnySwitch2": PrivateAnySwitch,
    "AnySwitch4": PrivateAnySwitch4,
    "DumpToConsole": DumpToConsole,
    "CLIPDistance": CLIPDistance,
}
