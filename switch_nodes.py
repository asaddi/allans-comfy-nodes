# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import torch

from .utils import PromptUtils

from comfy_execution.graph import ExecutionBlocker


class GenericSwitch:
    # Being server-side-only, it means we have a fixed number of inputs
    NUM_INPUTS = 2
    INPUT_TYPE = "MODEL"
    INPUT_NAME = "model"
    TITLE = "Model Switch 2"
    # NB Subclasses must re-define this as well
    RETURN_TYPES = (INPUT_TYPE,)

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

        for index in range(cls.NUM_INPUTS):
            d["optional"][f"{cls.INPUT_NAME}{index}"] = (cls.INPUT_TYPE,)

        return d

    FUNCTION = "switch"

    CATEGORY = "private/switch"

    def switch(self, unique_id, **kwargs):
        for index in range(self.NUM_INPUTS):
            input = kwargs.get(f"{self.INPUT_NAME}{index}")
            if input is not None:
                return (input,)
        return (ExecutionBlocker(None),)


class VAESwitch2(GenericSwitch):
    INPUT_TYPE = "VAE"
    INPUT_NAME = "vae"
    TITLE = "VAE Switch 2"
    RETURN_TYPES = (INPUT_TYPE,)


class ImageSwitch2(GenericSwitch):
    INPUT_TYPE = "IMAGE"
    INPUT_NAME = "image"
    TITLE = "Image Switch 2"
    RETURN_TYPES = (INPUT_TYPE,)


class ImageMaskSwitch:
    NUM_INPUTS = 2

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }
        for index in range(cls.NUM_INPUTS):
            d["optional"][f"image{index}"] = ("IMAGE",)
            d["optional"][f"mask{index}"] = (
                "MASK",
                {
                    "lazy": True,
                },
            )
        return d

    TITLE = "Image/Mask Switch 2"

    RETURN_TYPES = ("IMAGE", "MASK")

    FUNCTION = "run"

    CATEGORY = "private/switch"

    def check_lazy_status(self, unique_id, prompt, **kwargs):
        needed = []
        # If an image is present, make sure the associated mask is evaluated
        pu = PromptUtils(prompt)
        for index in range(self.NUM_INPUTS):
            image = kwargs.get(f"image{index}")
            mask_input = f"mask{index}"
            mask = kwargs.get(mask_input)
            if (
                image is not None
                and mask is None
                and pu.is_input_connected(unique_id, mask_input)
            ):
                needed.append(mask_input)

            if image is not None:
                # No later images will be selected, so might as well skip
                # their masks too.
                break

        return needed

    def run(self, unique_id, prompt, **kwargs):
        # The LoadImage node returns an empty/default mask [1,64,64] if it
        # loads an image without an alpha. We'll do the same whenever the
        # mask is missing.
        for index in range(self.NUM_INPUTS):
            image = kwargs.get(f"image{index}")
            mask = kwargs.get(f"mask{index}")
            if image is not None:
                return (image, mask if mask is not None else torch.zeros((1, 64, 64)))
        return (
            ExecutionBlocker(None),
            torch.zeros((1, 64, 64)),
        )


class ImageMaskSwitch4(ImageMaskSwitch):
    NUM_INPUTS = 4

    TITLE = "Image/Mask Switch 4"


NODE_CLASS_MAPPINGS = {
    "ModelSwitch2": GenericSwitch,
    "VAESwitch2": VAESwitch2,
    "ImageSwitch2": ImageSwitch2,
    "ImageMaskSwitch2": ImageMaskSwitch,
    "ImageMaskSwitch4": ImageMaskSwitch4,
}
