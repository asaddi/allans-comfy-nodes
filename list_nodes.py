from itertools import zip_longest
from random import Random

import torch


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

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "repeat"

    CATEGORY = "private/list"

    def repeat(self, image: list[torch.Tensor], amount: list[int]):
        result: list[torch.Tensor] = []
        for img, amt in zip_longest(image, amount):
            if img is None:
                img = image[-1]
            if amt is None:
                amt = amount[-1]

            for _ in range(amt):
                # Ensure each tensor is a copy
                result.append(torch.clone(img))

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

    INPUT_IS_LIST = True

    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "repeat"

    CATEGORY = "private/list"

    def repeat(self, samples: list[torch.Tensor], amount: list[int]):
        result: list[torch.Tensor] = []
        for latents, amt in zip_longest(samples, amount):
            if latents is None:
                latents = samples[-1]
            if amt is None:
                amt = amount[-1]

            lat = latents["samples"]
            for _ in range(amt):
                # Ensure each tensor is a copy
                result.append({"samples": torch.clone(lat)})

        return (result,)


class SeedList:
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
                        "min": cls.SEED_MIN,
                        "max": cls.SEED_MAX,
                        "default": 0,
                    },
                ),
            },
            "optional": {
                "input_list": ("*",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        for i in input_types:
            # Is seed an input?
            received_type = i.get("seed")
            if received_type is not None:
                # Validate its type
                if received_type != "INT":
                    return f"seed, {received_type} != INT"
        return True

    TITLE = "Seed List"

    INPUT_IS_LIST = True

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "run"

    CATEGORY = "private/list"

    def run(self, seed, input_list=None):
        # print(seed, None if input_list is None else len(input_list))
        if input_list is None:
            input_list = [None]  # Run at least once

        result = []
        for s, _ in zip_longest(seed, input_list):
            if s is not None:
                # Only re-seed if we actually have a seed i.e. not past
                # the end of the list
                self._random.seed(s)

            result.append(self._random.randint(self.SEED_MIN, self.SEED_MAX))

        return (result,)


NODE_CLASS_MAPPINGS = {
    "ImageList": ImageList,
    "LatentList": LatentList,
    "SeedList": SeedList,
}
