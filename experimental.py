# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import itertools
from pathlib import Path
from pprint import pprint
import random
import re

import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from .utils import AnyType, ComboType

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder


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


class MakeImageGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cell_width": (
                    "INT",
                    {
                        "min": 1,
                        "default": 128,
                    },
                ),
                "cell_height": (
                    "INT",
                    {
                        "min": 1,
                        "default": 128,
                    },
                ),
                "columns": (
                    "INT",
                    {
                        "min": 1,
                        "default": 5,
                    },
                ),
                "rows": (
                    "INT",
                    {
                        "min": 1,
                        "default": 5,
                    },
                ),
                "major": (
                    "BOOLEAN",
                    {
                        "label_on": "row_major",
                        "label_off": "col_major",
                        "default": True,
                    },
                ),
            }
        }

    TITLE = "Make Image Grid"

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "make_grid"

    CATEGORY = "private/image"

    def make_grid(
        self,
        image: list[torch.Tensor],
        cell_width: list[int],
        cell_height: list[int],
        columns: list[int],
        rows: list[int],
        major: list[bool],
    ):
        # Same disclaimer as my other INPUT_IS_LIST nodes
        # I'm going to assume the simple case until otherwise needed
        cell_width = cell_width[0]
        cell_height = cell_height[0]
        columns: int = columns[0]
        rows: int = rows[0]
        major: bool = major[0]

        input_images = []
        for batched_image in image:
            for img in batched_image:
                # TODO I'm just gonna assume it's already [0, 1] float32
                height, width, channels = img.shape

                if channels == 1:
                    # Convert to RGB if needed (will we ever get single-channel?)
                    img = img.expand(-1, -1, 3)
                if channels < 4:
                    # Add alpha channel
                    alpha = torch.ones((height, width, 1), dtype=torch.float32)
                    img = torch.cat((img, alpha), dim=2)

                # Center image on a square
                longest = max(height, width)
                centered = torch.zeros((longest, longest, 4), dtype=torch.float32)
                top = (longest - height) // 2
                left = (longest - width) // 2
                centered[top : top + height, left : left + width, :] = img

                # Currently have square image [H,W,C], C=4
                centered = centered.permute(2, 0, 1).unsqueeze(
                    0
                )  # First convert to [B,C,H,W]
                # Then resize to target
                new_image = F.interpolate(
                    centered, (cell_height, cell_width), mode="bilinear"
                )

                input_images.append(new_image[0])  # Back to [C,H,W]

        # Yes, I know of torchvision's make_grid
        # For now, we want column-major as an option

        output_images = []
        # Ooh, itertools.batched is Python 3.12+ only. FIXME?
        for batch in itertools.batched(input_images, rows * columns):
            # NB At the moment we don't pad between images
            grid = torch.zeros((4, cell_height * rows, cell_width * columns))
            for idx, img in enumerate(batch):
                if major:  # row-major
                    row = idx // columns
                    col = idx % columns
                else:
                    row = idx % rows
                    col = idx // rows

                # Place onto grid
                grid[
                    :,
                    row * cell_height : (row + 1) * cell_height,
                    col * cell_width : (col + 1) * cell_width,
                ] = img

            grid = grid.permute(1, 2, 0).unsqueeze(0)  # To [B,H,W,C]
            output_images.append(grid)

        return (output_images,)


class WriteTextImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "value": (
                    "*",
                    {
                        "forceInput": True,
                    },
                ),
                "write_to": (["alpha", "image"],),
                # Corner too, but for now we'll just force bottom-right.
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # TODO Check image's type
        return True

    TITLE = "Write Text to Image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "engrave"

    CATEGORY = "private/image"

    def _overlay_alpha(
        self, image: torch.Tensor, overlay: PIL.Image.Image
    ) -> torch.Tensor:
        """
        Convert the overlay to a mask and use it to replace the alpha
        channel of all images.
        """
        # To tensor
        alpha = TVF.to_tensor(overlay)  # Should be [3,H,W]

        # Extract alpha channel from overlay
        # And make a batch of 1
        alpha = alpha[None, -1, None, :, :]

        # It becomes the original image's alpha channel.
        # (Discarding any previous alpha channel)
        alpha = alpha.expand(image.shape[0], -1, -1, -1)
        image = torch.cat((image[:, :3, :, :], alpha), dim=1)
        return image

    def _overlay_direct(
        self, image: torch.Tensor, overlay: PIL.Image.Image
    ) -> list[torch.Tensor]:
        """
        Convert the overlay to a mask and use it to composite a solid color
        onto all images.
        """
        # To tensor
        alpha = TVF.to_tensor(overlay)  # Should be [4,H,W]
        overlay_image = alpha.unsqueeze(0)

        # Extract alpha channel from overlay
        # And make a batch of 1
        alpha = overlay_image[:, -1, None, :, :]
        overlay_image = overlay_image[:, :3, :, :]

        orig_alpha = None
        if image.shape[1] == 4:
            # Preserve original alpha and trim down to RGB
            orig_alpha = image[:, -1, None, :, :]
            image = image[:, :3, :, :]

        # Composite overlay over image
        image = image * alpha + overlay_image * (1.0 - alpha)

        # Restore original alpha, if needed
        if orig_alpha is not None:
            image = torch.cat((image, orig_alpha), dim=1)

        return image

    def engrave(self, image: torch.Tensor, value, write_to: str):
        # Convert to [B,C,H,W]
        image = image.permute(0, 3, 1, 2)

        # Convert value to text, keep 1st line only
        text = str(value).split("\n", 1)[0]

        # Scale things according to the image height
        text_height = image.shape[2] * 0.05
        text_pad = image.shape[2] * 0.025
        font = PIL.ImageFont.truetype(
            BASE_PATH / "calibrib.ttf", text_height
        )  # TODO font

        # Since we're dealing with the alpha channel, make initial overlay
        # fully transparent
        overlay_color = (0, 0, 0, 255)
        text_color = (0xCC, 0xFF, 0, 0)

        # Empty image of same size
        overlay = PIL.Image.new("RGBA", (image.shape[3], image.shape[2]), overlay_color)
        draw = PIL.ImageDraw.Draw(overlay)

        # Determine text size
        text_size = draw.textbbox((0, 0), text, font=font)[2:]

        # Write text
        draw.text(
            (
                overlay.size[0] - text_size[0] - text_pad,
                overlay.size[1] - text_size[1] - text_pad,
            ),
            text,
            font=font,
            fill=text_color,
        )

        if write_to == "alpha":
            result = self._overlay_alpha(image, overlay)
        elif write_to == "image":
            result = self._overlay_direct(image, overlay)
        else:
            raise ValueError(f"Unhandled write_to: {write_to}")

        # Back to [B,H,W,C]
        result = result.permute(0, 2, 3, 1)

        return (result,)


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
    "MakeImageGrid": MakeImageGrid,
    "WriteTextImage": WriteTextImage,
    # "ValueSubstitution": ValueSubstitution,
    "ImageRouter": ImageRouter,
    "RandomCombo2": RandomCombo,
    "MaskBlur": MaskBlur,
    "AnyRelay2": GenericRelay,
    # "AnySwitch2": PrivateAnySwitch,
    # "AnySwitch4": PrivateAnySwitch4,
    "DumpToConsole": DumpToConsole,
    # "CLIPDistance": CLIPDistance,
}
