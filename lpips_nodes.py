import lpips
import torch

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

    CATEGORY = "asaddi"

    def load(self, net: str):
        loss_fn = lpips.LPIPS(net=net, spatial=True)

        return (loss_fn,)


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    # Rescale image to [-1,1] as expected by LPIPS
    image = 2.0 * image - 1.0
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
            },
        }

    TITLE = "LPIPS Run"

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    FUNCTION = "calculate"

    CATEGORY = "asaddi"

    def calculate(
        self, lpips_model: lpips.LPIPS, reference: torch.Tensor, image: torch.Tensor
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
        # NB This is across the entire batch, so it will be wrong for batches
        image_loss = float(spatial_map.mean())
        print(f"image_loss = {image_loss:.3f}")

        # Convert spatial distance map [B,C,H,W] (C=1) to [B,H,W,C] (C=1)
        spatial_map = spatial_map.permute(0, 2, 3, 1).to(torch.float32)

        # Let's color it an obnoxious yellow for the hell of it
        # (TODO make it configurable?)
        # Apparently #ccff00 is a popular consensus for fluorescent yellow
        color = torch.tensor([0xCC / 255.0, 1.0, 0.0])
        spatial_map = color * spatial_map
        # It should now be [B,H,W,C] (C=3)

        # TODO include image_loss in outputs?
        return {"ui": {"image_loss": (image_loss,)}, "result": (spatial_map,)}


NODE_CLASS_MAPPINGS = {
    "LPIPSModel": LPIPSModel,
    "LPIPSRun": LPIPSRun,
}
