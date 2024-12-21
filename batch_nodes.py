# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import json
import os
from pathlib import Path
from typing import Any

import PIL
import PIL.Image
import numpy as np
import torch

from .utils import WorkflowUtils


class BatchImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING",),
                # Recurse?
                # Reverse?
                # Order by mtime?
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
    RETURN_NAMES = ("IMAGE", "MASK", "basename", "json")
    OUTPUT_IS_LIST = (True, True, True, True)

    FUNCTION = "batch_load"

    CATEGORY = "private/image"

    @staticmethod
    def build_file_list(path: Path) -> list[str]:
        to_process = []
        for root, dirs, files in os.walk(path):
            # For now, we aren't recursing
            dirs[:] = []

            # TODO filter by extension?
            to_process.extend([os.path.join(root, fn) for fn in files])

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
        out_basenames: list[str] = []
        out_json: list[Any] = []

        file_list = BatchImageLoader.build_file_list(path)

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

        for fn in file_list[start_file : start_file + max_files_per_run]:
            input_fn = Path(fn)

            im = PIL.Image.open(input_fn)
            try:
                a = np.asarray(im)
                json_metadata = type(self).extract_json(im, input_fn.name)
            finally:
                im.close()

            # TODO Will it always be [0, 255]?
            img = torch.tensor(a, dtype=torch.float32) / 255.0
            # Tensor will be [H,W,C] (always?)

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

            out_basenames.append(input_fn.stem)
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

        return (out_images, out_masks, out_basenames, out_json)


NODE_CLASS_MAPPINGS = {
    "BatchImageLoader": BatchImageLoader,
}
