# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
from itertools import zip_longest
import math
from random import Random

import torch

from .utils import ComboType

import folder_paths


def do_repeat(sequence: list, amount: int, repeat: str) -> list:
    """
    Repeats elements in a sequence based on the specified amount and repeat type.

    Args:
        sequence (list): The input sequence to be repeated.
        amount (int): The number of times each element or the entire sequence should be repeated.
        repeat (str): The type of repetition, either "consecutively" or "sequence".

    Returns:
        list: The resulting list after repeating the elements or the entire sequence.

    Raises:
        AssertionError: If the amount is less than 1 or the repeat type is not valid.
    """
    assert amount >= 1
    assert repeat in ("consecutively", "sequence")

    if repeat == "consecutively":
        # repeat each element before moving on to next
        result = []
        for ele in sequence:
            result.extend([ele] * amount)
    else:
        # repeat entire list
        result = sequence * amount

    return result


class ModelSequenceList:
    NUM_INPUTS = 2

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "required": {},
        }

        models = folder_paths.get_filename_list("checkpoints")
        for index in range(cls.NUM_INPUTS):
            # First one may not be "None"
            d["required"][f"model{index}"] = (
                models if index == 0 else ["None"] + models,
            )

        # I want these to appear in the UI after the models
        d["required"].update(
            {
                "amount": (
                    "INT",
                    {
                        "min": 1,
                        "default": 1,
                    },
                ),
                "repeat": (
                    [
                        "consecutively",
                        "sequence",
                    ],
                ),
            }
        )

        return d

    TITLE = "Model Sequence List 2"

    RETURN_TYPES = (ComboType("*"),)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "repeat"

    CATEGORY = "private/list"

    def repeat(self, amount: int, repeat: str, **kwargs):
        sequence = []
        for index in range(self.NUM_INPUTS):
            if (model := kwargs.get(f"model{index}")) != "None":
                sequence.append(model)

        return (do_repeat(sequence, amount, repeat),)


class StringSequenceList:
    NUM_INPUTS = 2

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "required": {
                "input0": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                    },
                ),
                "amount": (
                    "INT",
                    {
                        "min": 1,
                        "default": 1,
                    },
                ),
                "repeat": (
                    [
                        "consecutively",
                        "sequence",
                    ],
                ),
            },
            "optional": {},
        }

        # The rest of the inputs are optional
        for index in range(cls.NUM_INPUTS - 1):
            d["optional"][f"input{index + 1}"] = (
                "STRING",
                {
                    "multiline": True,
                    "forceInput": True,
                },
            )

        return d

    TITLE = "String Sequence List 2"

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "repeat"

    CATEGORY = "private/list"

    def repeat(self, input0: str, amount: int, repeat: str, **kwargs):
        sequence = [input0]
        for index in range(self.NUM_INPUTS - 1):
            if (input := kwargs.get(f"input{index + 1}")) is not None:
                sequence.append(input)

        return (do_repeat(sequence, amount, repeat),)


class StringSequenceList5(StringSequenceList):
    NUM_INPUTS = 5
    TITLE = "String Sequence List 5"


class RepeatStringList:
    INPUT_TYPE = "STRING"
    TITLE = "Repeat String List"
    # The following is evaluated only once, subclasses must redefine
    RETURN_TYPES = (INPUT_TYPE,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (
                    cls.INPUT_TYPE,
                    {
                        "forceInput": True,
                    },
                ),
                "amount": (
                    "INT",
                    {
                        "min": 1,
                        "default": 1,
                    },
                ),
                "repeat": (
                    [
                        "consecutively",
                        "sequence",
                    ],
                ),
            }
        }

    INPUT_IS_LIST = True

    OUTPUT_IS_LIST = (True,)

    FUNCTION = "repeat"

    CATEGORY = "private/list"

    def repeat(self, input: list, amount: list[int], repeat: list[str]):
        amount = amount[0]
        repeat = repeat[0]

        return (do_repeat(input, amount, repeat),)


class RepeatIntList(RepeatStringList):
    INPUT_TYPE = "INT"
    TITLE = "Repeat Int List"
    RETURN_TYPES = (INPUT_TYPE,)


class RepeatFloatList(RepeatStringList):
    INPUT_TYPE = "FLOAT"
    TITLE = "Repeat Float List"
    RETURN_TYPES = (INPUT_TYPE,)


class FloatListStepSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "end": (
                    "FLOAT",
                    {
                        "default": 0.1,
                    },
                ),
                "step_size": (
                    "FLOAT",
                    {
                        "default": 0.1,
                    },
                ),
            }
        }

    TITLE = "Float List (step size)"

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "run"

    CATEGORY = "private/list"

    def run(self, start, end, step_size):
        # Ensure sanity
        step_size = math.copysign(step_size, end - start)
        if math.isclose(step_size, 0.0):
            raise ValueError("step_size cannot be zero")

        step_count = math.ceil(round((end - start) / step_size, 3))

        result = []
        for i in range(step_count):
            # Feels saner to do it this way
            result.append(round(start + i * step_size, 3))

        # And the final step
        result.append(end)

        return (result,)


class ListCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    TITLE = "List Counter"

    INPUT_IS_LIST = True

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "count"

    CATEGORY = "private/list"

    def count(self, input, unique_id: list[str]):
        unique_id: str = unique_id[0]

        c = len(input)
        print(f"ListCounter #{unique_id} = {c}")
        return {"ui": {"count": (c,)}}


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
    # "ModelSequenceList2": ModelSequenceList,
    "StringSequenceList2": StringSequenceList,
    "StringSequenceList5": StringSequenceList5,
    "RepeatStringList": RepeatStringList,
    "RepeatFloatList": RepeatFloatList,
    "RepeatIntList": RepeatIntList,
    "ListCounter": ListCounter,
    "FloatList": FloatList,
    "FloatListStepSize": FloatListStepSize,
    "ImageList": ImageList,
    "LatentList": LatentList,
    "SeedList": SeedList,
}
