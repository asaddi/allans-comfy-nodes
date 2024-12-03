from pprint import pprint
import random
import re

import numpy as np
import torch

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder


class ValueSubstitution:
    """
    Currently used to add a list value (e.g. from FloatList) to filenames.

    The "%" subsitutution that occurs in the frontend only appears to occur
    once and does not work for lists.

    Images going into the save node should be re-batched to 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "*",
                    {
                        "forceInput": True,
                    },
                ),
                "template": (
                    "STRING",
                    {
                        "default": "ComfyUI-{value}",
                    },
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # TODO Check template's type
        return True

    TITLE = "Value Substitution"

    RETURN_TYPES = ("STRING",)

    FUNCTION = "render"

    CATEGORY = "private/list"

    def render(self, value, template: str):
        text = template.replace("{value}", str(value))
        return (text,)


class AnyType(str):
    def __ne__(self, other):
        if self == "*":
            # If we're the wildcard, match anything
            return False
        return super().__ne__(other)


class GenericRelay:
    NUM_INPUTS = 2

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "required": {
                "control": (
                    "INT",
                    {
                        "min": 0,
                        "max": cls.NUM_INPUTS - 1,
                        "default": 0,
                    },
                ),
            },
            "optional": {},
        }

        for index in range(cls.NUM_INPUTS):
            d["optional"][f"input{index}"] = (
                "*",
                {
                    "lazy": True,
                },
            )

        return d

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # We've had luck using this to prevent validation errors on our
        # wildcard inputs, so we'll just stick with it.
        # (As opposed to using the str subclass, as done below.)
        return True

    TITLE = "Any Relay 2"

    RETURN_TYPES = (AnyType("*"),)

    FUNCTION = "switch"

    CATEGORY = "private/switch"

    def check_lazy_status(self, control, **kwargs):
        # Only need the one chosen by control
        input_name = f"input{control}"
        input = kwargs.get(input_name)
        if input is None:
            # NB We don't check if the input is actually connected.
            # So this should actually output an error if there's nothing there
            # (the desired behavior)
            return [input_name]
        return []

    def switch(self, control, **kwargs):
        input = kwargs.get(f"input{control}")
        return (input,)


class ImageRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tag_probs": ("TAGPROBS",),
                "match_regexp": (
                    "STRING",
                    {
                        "default": "tag1|tag2",
                    },
                ),
                "func": (["argmax", "threshold"],),
                "arg": (
                    "FLOAT",
                    {
                        "default": 0.35,
                    },
                ),
            },
        }

    TITLE = "Image Router"

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("pos_match", "neg_match")

    FUNCTION = "route"

    CATEGORY = "private/switch"

    def proc_argmax(self, tag_probs: dict[str, float], arg: float) -> list[str]:
        tags = list(tag_probs.keys())
        probs = np.array(list(tag_probs.values()))
        # We want all indices, not just the first
        # So ironically, we can't use argmax
        max_prob = np.max(probs)
        max_indices = np.where(probs == max_prob)[0]
        matched = [tags[i] for i in max_indices]
        return matched

    def proc_threshold(self, tag_probs: dict[str, float], arg: float) -> list[str]:
        tags = list(tag_probs.keys())
        probs = np.array(list(tag_probs.values()))
        indices = np.where(probs > arg)[0]
        matched = [tags[i] for i in indices]
        return matched

    def route(
        self,
        image: torch.Tensor,
        tag_probs: dict[str, float],
        match_regexp: str,
        func: str,
        arg: float,
    ):
        funcs = {
            "argmax": self.proc_argmax,
            "threshold": self.proc_threshold,
        }
        tags = funcs[func](tag_probs, arg)
        print(f"resolved tags = {tags}")

        matcher = re.compile(match_regexp)
        matched = any([matcher.fullmatch(tag) is not None for tag in tags])

        # Ah crap, how will this work???
        # Hopefully a silent ExecutionBlocker does what we need...
        # Well, it does work, but it's apparently discouraged. How
        # else would we do it? Two string outputs (to be used as save prefix?)
        return (
            image if matched else ExecutionBlocker(None),
            ExecutionBlocker(None) if matched else image,
        )


class ComboType(str):
    def __ne__(self, other):
        if self == "*" and isinstance(other, list):
            return False
        return True


class RandomCombo:
    NUM_COMBOS = 2

    _random = random.Random()

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "required": {
                "seed": (
                    "INT",
                    {
                        "min": 0,
                        "max": 0xFFFFFFFF_FFFFFFFF,
                        "default": 0,
                    },
                ),
            },
        }
        for index in range(cls.NUM_COMBOS):
            d["required"][f"choice{index}"] = (
                "STRING",
                {"default": f"choice-{index + 1}"},
            )
            step = 1.0 / cls.NUM_COMBOS
            d["required"][f"threshold{index}"] = (
                "FLOAT",
                {
                    "min": 0.0,
                    "max": 1.0,
                    "default": round(step + index * step, 3),
                },
            )
        return d

    TITLE = "Random Combo 2"

    RETURN_TYPES = (ComboType("*"),)

    FUNCTION = "choose"

    CATEGORY = "private/switch"

    def choose(self, seed, **kwargs):
        # I don't know if it's possible to read the COMBO choices without
        # doing frontend stuff... which we want to avoid, because we want
        # this node usable in API.

        # So the onus is on the user to set correct values
        choices: tuple[float, str] = []
        for index in range(self.NUM_COMBOS):
            choice: str = kwargs.get(f"choice{index}")
            thresh: float = kwargs.get(f"threshold{index}")
            if choice and thresh is not None:  # Can they be empty??
                choices.append((thresh, choice))

        choices.sort(key=lambda x: x[0])

        # This feels dumb, but we'll go with it for now.
        # Other option is to convert seed to a float somehow.
        self._random.seed(seed)
        num = self._random.random()

        # Garbage in, garbage out.
        final_choice = choices[-1][1]
        for thresh, choice in choices:
            if num <= thresh:
                final_choice = choice
                break

        return (final_choice,)


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
    "ValueSubstitution": ValueSubstitution,
    "ImageRouter": ImageRouter,
    "RandomCombo2": RandomCombo,
    "MaskBlur": MaskBlur,
    "AnyRelay2": GenericRelay,
    # "AnySwitch2": PrivateAnySwitch,
    # "AnySwitch4": PrivateAnySwitch4,
    "DumpToConsole": DumpToConsole,
    # "CLIPDistance": CLIPDistance,
}
