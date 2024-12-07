# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import lpips
import matplotlib as mpl
import torch

from .utils import WorkflowUtils

from comfy.model_management import get_torch_device, unet_offload_device


class LPIPSModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "net": (["alex", "vgg"],),
            },
        }

    TITLE = "LPIPS Model"

    RETURN_TYPES = ("LPIPS_MODEL",)

    FUNCTION = "load"

    CATEGORY = "private/lpips"

    def load(self, net: str):
        loss_fn = lpips.LPIPS(net=net, spatial=True)

        return (loss_fn,)


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    # Note: With the "relative" switch on, tiny errors are being amplified.
    # In reality, images would be closer in 24-bit RGB space.
    # Convert image to 8-bit per channel first, then back.
    image = (255.0 * image).clamp(0, 255).to(torch.uint8)
    # Rescale image to [-1,1] as expected by LPIPS
    image = 2.0 * image.to(torch.float32) / 255.0 - 1.0
    # [B,H,W,C] -> [B,C,H,W]
    image = image.permute(0, 3, 1, 2)
    return image


class LPIPSRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lpips_model": ("LPIPS_MODEL",),
                "reference": ("IMAGE",),
                "image": ("IMAGE",),
                "relative": (
                    "BOOLEAN",
                    {
                        "label_on": "relative",
                        "label_off": "absolute",
                        "default": True,
                    },
                ),
                "grayscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    TITLE = "LPIPS Image Compare"

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "loss")
    OUTPUT_NODE = True

    FUNCTION = "calculate"

    CATEGORY = "private/lpips"

    def calculate(
        self,
        lpips_model: lpips.LPIPS,
        reference: torch.Tensor,
        image: torch.Tensor,
        relative: bool,
        grayscale: bool,
        unique_id: str,
        extra_pnginfo: dict,
    ):
        torch_device = get_torch_device()
        offload_device = unet_offload_device()

        if reference.shape[1:2] != image.shape[1:2]:
            raise ValueError("reference and image must have identical dimensions")

        reference = prepare_image(reference)
        image = prepare_image(image)

        lpips_model = lpips_model.to(torch_device)
        reference = reference.to(torch_device)
        image = image.to(torch_device)

        # TODO batching is going to be crazy. Just do cross product of
        # reference batches X image batches?!
        # Actually, what does lpips do with batches under the hood?!
        # A: As long as reference's B=1, it seems to do the right thing.
        # All bets are off when reference is batched, what is it doing?
        spatial_map: torch.Tensor = lpips_model(reference, image)

        lpips_model.to(offload_device)

        spatial_map = spatial_map.cpu()
        # FIXME This is across the entire batch, so it will be wrong for batches
        image_loss = float(spatial_map.mean())
        print(f"image_loss = {image_loss:.6f}")

        # Convert spatial distance map [B,C,H,W] (C=1) to [B,H,W,C] (C=1)
        spatial_map = spatial_map.permute(0, 2, 3, 1).to(torch.float32)

        if relative:
            spatial_map = (spatial_map - spatial_map.min()) / (
                spatial_map.max() - spatial_map.min()
            )

        if grayscale:
            spatial_map = spatial_map.expand(-1, -1, -1, 3)
        else:
            cmap = mpl.colormaps["inferno"]
            # And now to [B,H,W] for matplotlib
            spatial_map = spatial_map.reshape(spatial_map.shape[:3])
            spatial_map = cmap(spatial_map)[:, :, :, :3]  # Get rid of alpha channel
            # I think cmap converts it to an ndarray, convert it back
            spatial_map = torch.tensor(spatial_map, dtype=torch.float32)

        # It should now be [B,H,W,C] (C=3)

        # Also ensure outgoing metadata has record of image_loss
        wfu = WorkflowUtils(extra_pnginfo)
        wfu.set_property(unique_id, "loss", image_loss)

        return {
            "ui": {"image_loss": (image_loss,)},
            "result": (spatial_map, image_loss),
        }


NODE_CLASS_MAPPINGS = {
    "LPIPSModel": LPIPSModel,
    "LPIPSRun": LPIPSRun,
}
