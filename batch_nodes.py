# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import itertools
import json
import os
from pathlib import Path, PurePath
from typing import Any
import zipfile

import PIL
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional as TVF

from .utils import WorkflowUtils

import folder_paths


# I'm not sure why I'm putting this node in this module, but we'll go with
# it for now. Arguably, it's batch-like as well...
class SaveComicBookArchive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Metadata via (optional) JSON? Perhaps when loading batched
                # images?
                # Note: I don't think %-based substitution is available by
                # default? (Needs property set on frontend)
                # Might have to do it server-side as well (similar to updated
                # SaveImage node) for compatibility with API. TODO
                "archive_name": (
                    "STRING",
                    {
                        "default": "MyComicBook",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "img_",
                    },
                ),
                "image_format": (["jpeg", "png"],),
                # TODO Alternate way of handling this is having format-specific
                # nodes and an optional input.
                "image_quality": (
                    "INT",
                    {
                        "min": 0,
                        "max": 100,
                        "default": 85,
                        "tooltip": "Compression quality, typically 5-95. Only used for JPEG images.",
                    },
                ),
                # Append flag? For now, we will always append.
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # Note: The only other archive format we have ready access to (without
    # pulling in extra dependencies) is TAR (.cbt). I can't imagine it's
    # used much, especially over ZIP.
    TITLE = "Save ComicBookArchive (ZIP)"

    INPUT_IS_LIST = True

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "save"

    CATEGORY = "private/image"

    @staticmethod
    def get_counter(filename_prefix: str, infos: list[zipfile.ZipInfo]) -> int:
        # Note: We're allowing the filename_prefix to contain subdirectories.
        # We will canonicalize all filenames "as_posix" i.e. forward-slash.
        prefix = PurePath(filename_prefix).as_posix()
        prefix_len = len(prefix)
        # TODO optimize, this feels more complex than it needs to be
        paths = [
            # Only interested in existing files within the archive.
            # Also remove the extension.
            PurePath(info.filename).with_suffix("").as_posix()
            for info in infos
            if not info.is_dir()
        ]
        filtered = [p[prefix_len:] for p in paths if p.startswith(prefix)]
        counts: list[int] = []
        for p in filtered:
            try:
                count = int(p)
            except ValueError:
                count = 0
            counts.append(count)
        if not counts:
            # In case of no matches...
            counts = [0]
        return max(counts) + 1

    def save(
        self,
        image: list[torch.Tensor],
        archive_name: list[str],
        filename_prefix: list[str],
        image_format: list[str],
        image_quality: list[int],
        prompt: list[dict],
        extra_pnginfo: list[dict],
    ):
        archive_name: str = archive_name[0]
        filename_prefix: str = filename_prefix[0]
        image_format: str = image_format[0]
        image_quality: int = image_quality[0]

        _, ext = os.path.splitext(archive_name)
        if ext.lower() != ".cbz":
            archive_name += ".cbz"

        # Note: Default compression is just "store"
        # Give user a choice? Or just not bother? Going with latter...
        with zipfile.ZipFile(
            os.path.join(folder_paths.output_directory, archive_name), "a"
        ) as myzip:
            counter = SaveComicBookArchive.get_counter(
                filename_prefix, myzip.infolist()
            )
            filename_template = PurePath(filename_prefix).as_posix() + "{:05d}.{}"
            # print(f"counter = {counter}")
            # print(f"filename_template = {filename_template}")

            for batched, pr, ex in itertools.zip_longest(image, prompt, extra_pnginfo):
                if batched is None:
                    # No point if there aren't anymore images
                    break
                if pr is None:
                    pr = prompt[-1]
                if ex is None:
                    ex = extra_pnginfo[-1]

                for img in batched:
                    # TODO deal with other channel configurations
                    im = TVF.to_pil_image(img.permute(2, 0, 1))
                    if image_format == "png":
                        with myzip.open(
                            filename_template.format(counter, "png"), "w"
                        ) as out:
                            # TODO metadata
                            im.save(out, "PNG")
                    elif image_format == "jpeg":
                        with myzip.open(
                            filename_template.format(counter, "jpg"), "w"
                        ) as out:
                            # TODO metadata
                            im.save(out, "JPEG", quality=image_quality, optimize=True)
                    else:
                        raise ValueError(f"Unhandled image_format: {image_format}")
                    counter += 1

        return ()


class BatchImageLoader:
    IMG_EXTENSIONS = set(
        [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING",),
                "recurse": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                # Reverse?
                # Order by mtime? Nah, that's asking for trouble.
                "max_batch_size": (
                    "INT",
                    {
                        "min": 1,
                        "max": 4096,
                        "default": 4,
                    },
                ),
                "start_file": (
                    "INT",
                    {
                        "min": 1,
                        "default": 1,
                    },
                ),
                "max_files_per_run": (
                    "INT",
                    {
                        "min": 1,
                        "default": 100,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, path):
        if not path:
            return "required"
        input_path = Path(path)
        if not input_path.is_dir():
            return "must be a directory"
        # Note: Don't really want to validate start_file here
        # In theory, we can though...
        return True

    @classmethod
    def IS_CHANGED(
        cls,
        path,
        recurse,
        max_batch_size,
        start_file,
        max_files_per_run,
        unique_id,
        extra_pnginfo,
    ):
        input_path = Path(path)
        return input_path.stat().st_mtime

    TITLE = "Batch Image Loader"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "path", "json")
    OUTPUT_IS_LIST = (True, True, True, True)

    FUNCTION = "batch_load"

    CATEGORY = "private/image"

    @staticmethod
    def build_file_list(path: Path | str, recurse: bool = False) -> list[Path]:
        to_process = []
        for root, dirs, files in Path(path).walk():
            if not recurse:
                dirs[:] = []

            to_process.extend(
                [
                    root / fn
                    for fn in files
                    if os.path.splitext(fn)[1].lower()
                    in BatchImageLoader.IMG_EXTENSIONS
                ]
            )

        return sorted(to_process)

    @classmethod
    def extract_json(cls, image: PIL.Image.Image, filename: str) -> Any:
        # For now, only attempt to extract from PNGs
        if image.format != "PNG":
            return {}
        # Convert the top-level items to actual JSON
        out = {}
        for key, data in image.text.items():
            try:
                d = json.loads(data)
                out[key] = d
            except json.JSONDecodeError as e:
                print(f"{cls.__name__}: {filename}: Text key {key} not valid JSON")
                # Just store it as a string
                # (Note: It may be a str subclass, hence the cast)
                out[key] = str(data)
        return out

    def batch_load(
        self,
        path: str,
        recurse: bool,
        max_batch_size: int,
        start_file: int,
        max_files_per_run: int,
        unique_id: str,
        extra_pnginfo: dict,
    ):
        # I'll leave the UI at 1-based indexing to be "user friendly"
        start_file = start_file - 1

        out_images: list[torch.Tensor] = []
        out_masks: list[torch.Tensor] = []
        out_paths: list[str] = []
        out_json: list[Any] = []

        file_list = BatchImageLoader.build_file_list(path, recurse=recurse)

        if start_file >= len(file_list):
            raise ValueError("start_file exceeds number of files in directory")

        batch_dim: list[tuple[int, int]] = []
        batch_images: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor | None] = []

        def add_batch_to_output(force: bool = False):
            if not batch_dim:
                # Nothing to do
                return
            if len(batch_images) >= max_batch_size or force:
                # Create a new (tensor) batch
                out_images.append(torch.stack(batch_images))

                if any([x is not None for x in batch_masks]):
                    # Create empty mask of proper dimensions for any missing
                    # masks
                    masks = [
                        mask
                        if mask is not None
                        else torch.zeros(1, batch_dim[0][0], batch_dim[0][1])
                        for mask in batch_masks
                    ]
                    # Batch them up together
                    out_masks.append(torch.stack(masks))
                else:
                    # Create a batch of empty/default masks ([H,W] = [64, 64])
                    out_masks.append(torch.zeros(len(batch_images), 64, 64))

                # Zero out batch lists
                batch_dim[:] = []
                batch_images[:] = []
                batch_masks[:] = []

        for input_fn in file_list[start_file : start_file + max_files_per_run]:
            im = PIL.Image.open(input_fn)
            try:
                a = np.asarray(im)
                json_metadata = type(self).extract_json(im, input_fn.name)
            finally:
                im.close()

            # TODO Will it always be [0, 255]?
            img = torch.tensor(a, dtype=torch.float32) / 255.0
            # Tensor will be [H,W,C] (always?)

            if len(img.shape) < 3:
                # No, not always [H,W,C]
                img = img.unsqueeze(-1)

            if img.shape[2] < 3:
                # Grayscale. Convert to 3 channels.
                # TODO Is it valid to leave it at 1 channel?
                img = img.expand(-1, -1, 3)

            mask = None
            if img.shape[2] == 4:
                # Extract alpha channel as mask
                mask = 1.0 - img[:, :, 3]
                img = img[:, :, :3]

            # If we're building a batch and this new image is a different
            # size, force a new batch
            force_batch = batch_dim and (
                batch_dim[0][0] != img.shape[0] or batch_dim[0][1] != img.shape[1]
            )

            add_batch_to_output(force=force_batch)

            if not batch_dim:
                # Start of new batch
                batch_dim[:] = [(img.shape[0], img.shape[1])]
                # I'll trust that they've been emptied already
                assert len(batch_images) == 0
                assert len(batch_masks) == 0

            batch_images.append(img)
            batch_masks.append(mask)

            out_paths.append(str(input_fn))
            out_json.append(json_metadata)

        # Catch any stragglers
        add_batch_to_output(force=True)

        # Scrub the path from the outgoing metadata
        wfu = WorkflowUtils(extra_pnginfo)
        wfu.set_widget(unique_id, 0, "")
        # And update start_file
        wfu.set_widget(
            unique_id, 2, 1 + min(len(file_list), start_file + max_files_per_run)
        )

        return (out_images, out_masks, out_paths, out_json)


NODE_CLASS_MAPPINGS = {
    "SaveComicBookArchive": SaveComicBookArchive,
    "BatchImageLoader": BatchImageLoader,
}
