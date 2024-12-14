# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
from pathlib import Path
from pprint import pprint
import random

import torch
import torch.nn.functional as F

from .utils import AnyType, ComboType

from comfy_execution.graph import ExecutionBlocker


BASE_PATH = Path(__file__).parent.resolve()


# Make it look like Noise_EmptyNoise/Noise_RandomNoise
class Noise_MixedNoise:
    def __init__(
        self, noise_base, weight, noise_mix=None, mask: torch.Tensor | None = None
    ):
        self.noise_base = noise_base
        self.weight = weight
        self.noise_mix = noise_mix
        self.mask = mask

    # Apparently some samplers read the seed
    @property
    def seed(self) -> int:
        return self.noise_base.seed

    def generate_noise(self, input_latent):
        base = self.noise_base.generate_noise(input_latent)
        # If there's nothing else to do, do nothing
        if self.noise_mix is None or self.weight == 0.0:
            return base

        mix = self.noise_mix.generate_noise(input_latent)

        latent_image = input_latent["samples"]
        if self.mask is not None:
            # Mask is [B,H,W]. Make it match latent [B,C,H,W]
            # Note: Take the 1st mask TODO Is this correct?
            mask: torch.Tensor = self.mask[None, 0, None, :, :]
            # Ensure it's the same [H,W] as latent...
            mask = F.interpolate(mask, latent_image.shape[-2:], mode="bilinear")
            # Mask should now be [1,1,H,W]
        else:
            mask = torch.ones(
                1, 1, latent_image.shape[2], latent_image.shape[3], dtype=torch.float32
            )

        # And finally, apply it to the noise being mixed in...
        unmasked_base = base * (1.0 - mask)
        masked_base = base * mask
        mix = mix * mask

        # Mix in second noise according to weight
        mixed = masked_base + mix * self.weight
        # Note: mixed doesn't include anything outside the mask.
        # That's unmasked_base

        # Perform Z-score normalization on masked area. Thanks, Llama 3.3!
        new_std, new_mean = torch.std_mean(mixed)
        normalized = (mixed - new_mean) / new_std

        orig_std, orig_mean = torch.std_mean(masked_base)
        mixed = normalized * orig_std + orig_mean

        # Finally, add in the unmasked area (if any)
        mixed = unmasked_base + mixed

        # TODO The whole batch_index thing... do we have to worry?
        return mixed


class MixNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_base": ("NOISE",),
                "weight": (
                    "FLOAT",
                    {
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "default": 0.0,
                    },
                ),
            },
            "optional": {
                "noise_mix": (
                    "NOISE",
                    {
                        "lazy": True,
                    },
                ),
                "mask": ("MASK",),
            },
        }

    TITLE = "Mix Noise"

    RETURN_TYPES = ("NOISE",)

    FUNCTION = "get_noise"

    CATEGORY = "private/noise"

    def check_lazy_status(
        self,
        noise_base,
        weight: float,
        noise_mix=None,
        mask: torch.Tensor | None = None,
    ):
        needed = []
        # Pretty meager savings, but quick to check
        if weight != 0.0 and noise_mix is None:
            needed.append("noise_mix")
        # TODO Check mask too? But we'd have to see if it was connected first...
        return needed

    def get_noise(
        self,
        noise_base,
        weight: float,
        noise_mix=None,
        mask: torch.Tensor | None = None,
    ):
        return (Noise_MixedNoise(noise_base, weight, noise_mix=noise_mix, mask=mask),)


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
                "input": ("*",),
                "use_pprint": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "pprint",
                        "label_off": "raw",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    TITLE = "Dump To Console"

    RETURN_TYPES = (AnyType("*"),)
    RETURN_NAMES = ("passthrough",)
    OUTPUT_NODE = True

    FUNCTION = "run"

    CATEGORY = "private/debug"

    def run(self, input, use_pprint, unique_id):
        print(f"DumpToConsole (#{unique_id}) = ", end="")
        if use_pprint:
            pprint(input)
        else:
            print(input)

        return (input,)


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
    "MixNoise": MixNoise,
    # "ValueSubstitution": ValueSubstitution,
    "RandomCombo2": RandomCombo,
    # "AnyRelay2": GenericRelay,
    # "AnySwitch2": PrivateAnySwitch,
    # "AnySwitch4": PrivateAnySwitch4,
    "DumpToConsole": DumpToConsole,
    # "CLIPDistance": CLIPDistance,
}
