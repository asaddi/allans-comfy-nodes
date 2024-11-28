from itertools import zip_longest
from random import Random

import torch


class FloatList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "min": 1,
                        "default": 10,
                    },
                ),
            }
        }

    TITLE = "Float List"

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "run"

    CATEGORY = "private/list"

    def run(self, start, end, steps):
        interval = end - start
        steps = steps - 1
        result = []
        for i in range(steps):
            result.append(round(start + interval * i / steps, 3))
        # And the final step
        result.append(end)

        return (result,)


class ImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": (
                    "INT",
                    {
                        "min": 1,
                        "default": 1,
                    },
                ),
            }
        }

    TITLE = "Repeat Image List"

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "repeat"

    CATEGORY = "private/list"

    def repeat(self, image: torch.Tensor, amount: int):
        result: list[torch.Tensor] = []
        for _ in range(amount):
            # Ensure each tensor is a copy
            result.append(torch.clone(image))

        return (result,)


class LatentList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "amount": (
                    "INT",
                    {
                        "min": 1,
                        "default": 1,
                    },
                ),
            }
        }

    TITLE = "Repeat Latent List"

    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "repeat"

    CATEGORY = "private/list"

    def repeat(self, samples: dict, amount: int):
        result: list[dict] = []
        lat = samples["samples"]
        for _ in range(amount):
            # Ensure each tensor is a copy
            result.append({"samples": torch.clone(lat)})

        return (result,)


class SeedList:
    # Note: This is short of the full 64-bits to match JavaScript's integer
    # limitation.
    SEED_MIN = 0
    SEED_MAX = 2**53 - 1

    _random = Random()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        # Like most seeded nodes, accept the full 64-bit range
                        "min": 0,
                        "max": 0xFFFF_FFFF_FFFF_FFFF,
                        "default": 0,
                    },
                ),
                "amount": (
                    "INT",
                    {
                        "min": 1,
                        "default": 1,
                    },
                ),
            },
            "optional": {
                "input_list": ("*",),
            },
        }

    # TODO At this point, it's probably easier to create a str subclass
    # for the "*" type.
    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        for i in input_types:
            # Is seed an input?
            received_type = i.get("seed")
            if received_type is not None:
                # Validate its type
                if received_type != "INT":
                    return f"seed, {received_type} != INT"

            received_type = i.get("amount")
            if received_type is not None:
                if received_type != "INT":
                    return f"amount, {received_type} != INT"

        return True

    TITLE = "Seed List"

    # Note: This is True so we can distinguish the start of the list and
    # only set _random's seed when we have actual values, not repeated
    # values.
    INPUT_IS_LIST = True

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seeds",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "run"

    CATEGORY = "private/list"

    def run(self, seed, amount, input_list=None):
        # print(seed, None if input_list is None else len(input_list))
        if input_list is None:
            input_list = [None] * amount[0]

        result = []
        for s, _ in zip_longest(seed, input_list):
            if s is not None:
                # Only re-seed if we actually have a seed i.e. not past
                # the end of the list
                self._random.seed(s)

            result.append(self._random.randint(self.SEED_MIN, self.SEED_MAX))

        return (result,)


NODE_CLASS_MAPPINGS = {
    "FloatList": FloatList,
    "ImageList": ImageList,
    "LatentList": LatentList,
    "SeedList": SeedList,
}
