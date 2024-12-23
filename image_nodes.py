# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import itertools
import math
import os
from pathlib import Path
import re
import shutil
import uuid

from aiohttp import web
from aiohttp.web_request import Request
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder
import folder_paths
from server import PromptServer


BASE_PATH = Path(__file__).parent.resolve()


class ImageCropSquare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop": (["center", "top-left", "bottom-right"],),
            }
        }

    TITLE = "Image Crop Square"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "expand"

    CATEGORY = "private/image"

    def expand(self, image: torch.Tensor, crop: str):
        _, height, width, _ = image.shape
        shortest = min(height, width)
        if crop == "center":
            x = (width - shortest) // 2
            y = (height - shortest) // 2
        elif crop == "top-left":
            x, y = 0, 0
        elif crop == "bottom-right":
            x = width - shortest
            y = height - shortest
        else:
            raise ValueError(f"Unhandled crop: {crop}")

        graph = GraphBuilder()
        image_crop = graph.node(
            "ImageCrop", image=image, width=shortest, height=shortest, x=x, y=y
        )
        return {
            "result": (image_crop.out(0),),
            "expand": graph.finalize(),
        }


class ImageBuffer:
    _STORAGE_MAP: dict[str, Path] = {}

    def __init__(self):
        self._uuid = uuid.uuid4()
        # print(f"uuid = {self._uuid}")
        self._storage_path = Path(folder_paths.temp_directory) / str(self._uuid)
        self._storage_path.mkdir(exist_ok=True)
        self._counter = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "lazy": True,
                    },
                ),
                "action": (
                    "BOOLEAN",
                    {
                        "label_on": "acc",
                        "label_off": "rel",
                        "default": True,
                    },
                ),
                "start_index": (
                    "INT",
                    {
                        "min": -1,
                        "default": -1,
                    },
                ),
                # Rest of widgets will be added by frontend
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, image, action, start_index, unique_id):
        if not action[0]:
            storage_path = cls._STORAGE_MAP.get(unique_id[0])
            if storage_path is not None:
                return storage_path.stat().st_mtime
        return float("NaN")

    TITLE = "Image Buffer"

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    OUTPUT_NODE = True

    FUNCTION = "buffer"

    CATEGORY = "private/image"

    def _ensure_mapping(self, unique_id):
        unique_id = str(unique_id)
        mapping = type(self)._STORAGE_MAP
        # Maybe safer to force it every time?
        if unique_id not in mapping:
            mapping[unique_id] = self._storage_path

    @staticmethod
    def _build_file_list(storage_path: Path) -> list[str]:
        to_process = []
        for root, dirs, files in os.walk(storage_path):
            # No recursion
            dirs[:] = []

            # Filter by extension?
            to_process.extend([os.path.join(root, fn) for fn in files])

        return sorted(to_process)

    def check_lazy_status(self, image, action, start_index, unique_id):
        self._ensure_mapping(unique_id[0])
        if action[0] and image[0] is None:
            return ["image"]
        else:
            return []

    def _trim_file_list(self, index: int) -> list[str]:
        files = ImageBuffer._build_file_list(self._storage_path)

        if index > -1 and index < len(files):
            to_delete = files[index:]
            # print(f"to_delete = {to_delete}")
            files = files[:index]
            for fn in to_delete:
                Path(fn).unlink(missing_ok=True)
            # Maybe re-read the directory instead?

        return files

    def buffer(
        self,
        image: list[torch.Tensor],
        action: list[bool],
        start_index: list[int],
        unique_id,
    ):
        action: bool = action[0]
        start_index: int = start_index[0]
        unique_id: str = unique_id[0]

        self._ensure_mapping(unique_id)
        files = self._trim_file_list(start_index)
        if action:  # Accumulate
            added = 0
            for batched_image in image:
                for img in batched_image:
                    # img is [H,W,C]
                    d = {"image": img.contiguous()}
                    safetensors.torch.save_file(
                        d, self._storage_path / f"{self._counter:08d}.sft"
                    )
                    self._counter += 1
                    added += 1

            # TODO maybe it's saner to just re-read the directory
            return {
                "ui": {"image_count": (len(files) + added,)},
                "result": ([ExecutionBlocker(None)],),
            }
        else:  # Release
            if not files:
                raise ValueError("no images accumulated")

            images_out: list[torch.Tensor] = []
            for fn in files:
                sft = safetensors.safe_open(fn, "pt")
                t = sft.get_tensor("image")
                # They were saved [H,W,C]. Need to make into batches of 1.
                # TODO batching of like-sized images?
                images_out.append(t.unsqueeze(0))

            return {"ui": {"image_count": (len(files),)}, "result": (images_out,)}


@PromptServer.instance.routes.get("/image_buffer/{node_id}/count")
async def get_image_buffer_count(request: Request):
    node_id = request.match_info.get("node_id")
    if node_id:
        storage_path = ImageBuffer._STORAGE_MAP.get(node_id)
        if storage_path is not None:
            files = ImageBuffer._build_file_list(storage_path)
            return web.json_response([len(files)])
    return web.json_response(None)


@PromptServer.instance.routes.delete("/image_buffer/{node_id}")
async def clear_image_buffer(request: Request):
    node_id = request.match_info.get("node_id")
    if node_id:
        storage_path = ImageBuffer._STORAGE_MAP.get(node_id)
        if storage_path is not None:
            # Delete and remake the entire directory
            shutil.rmtree(storage_path, ignore_errors=True)
            storage_path.mkdir(exist_ok=True)
            return web.json_response(True)
    return web.json_response(False)


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
                    "BOOLEAN,FLOAT,INT,STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "write_to": (["alpha", "image"],),
                "corner": (["bottom-right", "bottom-left", "top-left", "top-right"],),
                "color": (
                    "INT",
                    {
                        "min": 0x00_00_00,
                        "max": 0xFF_FF_FF,
                        # Default to fluorescent yellow
                        "default": 0xCC_FF_00,
                    },
                ),
                # Font?
                # Text height?
                # Text padding?
                # Do we want to bother with all that? Not fond of nodes with
                # a million widgets.
            },
        }

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
        alpha = TVF.to_tensor(overlay)  # Should be [4,H,W]

        # Extract alpha channel from overlay
        # And make a batch of 1
        alpha = alpha[None, -1, None, :, :]
        # Repeat overlayed alpha to batch size
        alpha = alpha.expand(image.shape[0], -1, -1, -1)

        if image.shape[1] == 4:
            # Take original alpha from image
            orig_alpha = image[:, -1, None, :, :]
            # Merge overlayed alpha with original alpha
            # Basically, overlayed alpha is on top/has precedence
            alpha = torch.where(alpha == 1.0, orig_alpha, alpha)

        # And now it becomes the original image's alpha channel.
        image = torch.cat((image[:, :3, :, :], alpha), dim=1)
        return image

    def _overlay_direct(
        self, image: torch.Tensor, overlay: PIL.Image.Image
    ) -> list[torch.Tensor]:
        """
        Convert the overlay to a mask and use it to composite the text
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

    def engrave(
        self, image: torch.Tensor, value, write_to: str, corner: str, color: int
    ):
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

        # Make initial overlay fully opaque. In reality, the RGB channels
        # will be replaced by the original image.
        overlay_color = (0, 0, 0, 255)
        # Just in case we end up writing directly to the image, set the text
        # color. Make the text alpha fully transparent, regardless.
        text_color = ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF, 0)

        # Empty image of same size
        overlay = PIL.Image.new("RGBA", (image.shape[3], image.shape[2]), overlay_color)
        draw = PIL.ImageDraw.Draw(overlay)

        # Determine text size
        text_size = draw.textbbox((0, 0), text, font=font)[2:]

        if corner == "bottom-right":
            pos = (
                overlay.size[0] - text_size[0] - text_pad,
                overlay.size[1] - text_size[1] - text_pad,
            )
        elif corner == "bottom-left":
            pos = (text_pad, overlay.size[1] - text_size[1] - text_pad)
        elif corner == "top-left":
            pos = (text_pad, text_pad)
        elif corner == "top-right":
            pos = (overlay.size[0] - text_size[0] - text_pad, text_pad)
        else:
            raise ValueError(f"Unhandled corner: {corner}")

        # Write text
        draw.text(
            pos,
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("pos_match", "neg_match", "resolved_tags", "matched")

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
            ", ".join(tags),
            matched,
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
    "ImageCropSquare": ImageCropSquare,
    "ImageBuffer": ImageBuffer,
    "FlattenImageAlpha": FlattenImageAlpha,
    "MakeImageGrid": MakeImageGrid,
    "WriteTextImage": WriteTextImage,
    "ImageRouter": ImageRouter,
    "MaskBlur": MaskBlur,
}
