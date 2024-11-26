import torch
from comfy_execution.graph import ExecutionBlocker


class ImageMaskSwitch:
    NUM_INPUTS = 2

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
        for index in range(cls.NUM_INPUTS):
            d["optional"][f"image{index}"] = ("IMAGE",)
            d["optional"][f"mask{index}"] = ("MASK", {
                "lazy": True,
            })
        return d

    TITLE = "Image/Mask Switch 2"

    RETURN_TYPES = ("IMAGE", "MASK")

    FUNCTION = "run"

    CATEGORY = "private/switch"

    def check_lazy_status(self, unique_id, **kwargs):
        needed = []
        # If an image is present, make sure the associated mask is evaluated
        for index in range(self.NUM_INPUTS):
            image = kwargs.get(f"image{index}")
            mask_input = f"mask{index}"
            mask = kwargs.get(mask_input)
            if image is not None and mask is None:
                needed.append(mask_input)
        return needed

    def run(self, unique_id, **kwargs):
        for index in range(self.NUM_INPUTS):
            image = kwargs.get(f"image{index}")
            mask = kwargs.get(f"mask{index}")
            if image is not None:
                return (image, mask)
        return (
            ExecutionBlocker(f"Node #{unique_id}: All image inputs missing"),
            torch.zeros((1, 64, 64)),
        )


class ImageMaskSwitch4(ImageMaskSwitch):
    NUM_INPUTS = 4

    TITLE = "Image/Mask Switch 4"


NODE_CLASS_MAPPINGS = {
    "ImageMaskSwitch2": ImageMaskSwitch,
    "ImageMaskSwitch4": ImageMaskSwitch4,
}
