# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import itertools
import math
from pathlib import Path
import re

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder


BASE_PATH = Path(__file__).parent.resolve()


# Goes against my philosophy of avoiding nodes that can be done with Core
# nodes alone (and then not using node expansion).
# Hmm, but technically, this still needs a "Get Image Dim"-type node, which
# isn't part of Core...
class FlattenImageAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": (
                    "INT",
                    {
                        "min": 0x00_00_00,
                        "max": 0xFF_FF_FF,
                        "default": 0x00_00_00,
                    },
                ),
            }
        }

    TITLE = "Flatten Image Alpha"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "flatten"

    CATEGORY = "private/image"

    def flatten(self, image: torch.Tensor, color: int):
        batches, height, width, channels = image.shape
        if channels < 4:
            # No alpha channel, nothing to do. Just pass through.
            return (image,)

        # Separate image & alpha
        alpha = image[:, :, :, -1, None]
        image = image[:, :, :, :3]

        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF

        # Create the background
        bkg = torch.tile(
            torch.tensor([r, g, b], dtype=torch.float32) / 255.0, (height, width, 1)
        )
        bkg = bkg.expand(batches, -1, -1, -1)

        # Composite image onto background
        image = image * alpha + bkg * (1.0 - alpha)

        return (image,)


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
                "padding": (
                    "INT",
                    {
                        "min": 0,
                        "default": 2,
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
        padding: list[int],
    ):
        # Same disclaimer as my other INPUT_IS_LIST nodes
        # I'm going to assume the simple case until otherwise needed
        cell_width0 = cell_width[0]
        cell_height0 = cell_height[0]
        columns: int = columns[0]
        rows: int = rows[0]
        major: bool = major[0]
        padding: int = padding[0]

        # First pre-process all images, unbatching them and converting
        # to [H,W,C], C=4
        input_images: list[torch.Tensor] = []
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

                input_images.append(img)

        # Next, figure out cell size
        # for batch in itertools.batched(input_images, rows * columns):

        # Yes, I know of torchvision's make_grid
        # For now, we want column-major as an option

        def resized_dims(height, width) -> tuple[int, int]:
            if height == width:
                h = cell_height0
                w = cell_width0
            elif height > width:
                h = cell_height0
                w = math.ceil(cell_width0 * (width / height))
            elif width > height:
                h = math.ceil(cell_height0 * (height / width))
                w = cell_width0
            return h, w

        output_images: list[torch.Tensor] = []
        # Ooh, itertools.batched is Python 3.12+ only. FIXME?
        for batch in itertools.batched(input_images, rows * columns):
            # First, figure out cell size for this set of images
            cell_height: int = 0
            cell_width: int = 0
            new_batch: list[torch.Tensor] = []
            for img in batch:
                height, width, _ = img.shape

                h, w = resized_dims(height, width)
                cell_height = max(cell_height, h)
                cell_width = max(cell_width, w)

                # Next, resize image to fit in target cell size
                # Currently have image [H,W,C], C=4
                img = img.permute(2, 0, 1).unsqueeze(0)  # First convert to [B,C,H,W]
                # Then resize to target size
                img = F.interpolate(img, (h, w), mode="bilinear")

                new_batch.append(img[0])

            # print(f"cell dim: {cell_height} x {cell_width}")

            grid = torch.zeros(
                (
                    4,
                    padding + (cell_height + padding) * rows,
                    padding + (cell_width + padding) * columns,
                )
            )
            for idx, img in enumerate(new_batch):
                _, h, w = img.shape

                # Pad to cell size
                padded = torch.zeros((4, cell_height, cell_width), dtype=torch.float32)
                top = (cell_height - h) // 2
                left = (cell_width - w) // 2
                padded[:, top : top + h, left : left + w] = img

                if major:  # row-major
                    row = idx // columns
                    col = idx % columns
                else:
                    row = idx % rows
                    col = idx // rows

                # Then place onto grid
                top = padding + row * (cell_height + padding)
                left = padding + col * (cell_width + padding)
                grid[:, top : top + cell_height, left : left + cell_width] = padded

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


NODE_CLASS_MAPPINGS = {
    "FlattenImageAlpha": FlattenImageAlpha,
    "MakeImageGrid": MakeImageGrid,
    "WriteTextImage": WriteTextImage,
    "ImageRouter": ImageRouter,
    "MaskBlur": MaskBlur,
}
