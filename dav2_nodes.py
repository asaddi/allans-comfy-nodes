from pathlib import Path
import matplotlib as mpl
import torch

# Note: Current as of 7f2e0274
from .depth_anything_v2.dpt import DepthAnythingV2
from .models import ModelManager

from comfy.model_management import get_torch_device, unet_offload_device


MODELS = ModelManager(
    models_subdir="depth_anything_v2",
    allowed_files=["*.pth"],
    config_file="depth_anything_v2.override.yaml",
    default_config_file="depth_anything_v2.yaml",
)
MODELS.load()


model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


class DepthAnythingV2Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["vitl", "vitb", "vits"],),
            },
        }

    TITLE = "DepthAnythingV2 Model"

    RETURN_TYPES = ("DAV2MODEL",)

    FUNCTION = "load"

    CATEGORY = "private/image"

    def load(self, model):
        model_name = model
        model_path = Path(MODELS.download(model_name))

        model = DepthAnythingV2(**model_configs[model_name])
        model.load_state_dict(
            torch.load(
                model_path / f"depth_anything_v2_{model_name}.pth",
                weights_only=True,
                map_location="cpu",
            )
        )
        model.eval()

        return (model,)


class DepthAnythingV2Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dav2_model": ("DAV2MODEL",),
                "image": ("IMAGE",),
                "grayscale": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
        }

    TITLE = "DepthAnythingV2 Run"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "private/image"

    def run(self, dav2_model, image, grayscale):
        torch_device = get_torch_device()
        offload_device = unet_offload_device()

        # model expects 8-bit BGR
        image = (255.0 * image).clamp(0, 255).to(torch.uint8)
        # Switch from RGB to BGR
        image = image[:, :, :, [2, 1, 0]]

        dav2_model.to(torch_device)

        depth_maps = []
        for img in image:
            # Note: It expects a numpy array.
            # But it will convert it to a tensor and conditionally move
            # it to cuda/mps/cpu. So the model better be in the same place!
            depth = dav2_model.infer_image(img.numpy())

            # Note: And we get back a numpy array
            depth = torch.tensor(depth, dtype=torch.float32)
            # Rescale to [0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min())

            if grayscale:
                # From [H,W] to [H,W,C] where C=1
                depth = depth.unsqueeze(-1)
                # Repeat channel for grayscale
                depth = depth.expand(-1, -1, 3)
            else:
                cmap = mpl.colormaps["Spectral_r"]
                depth = cmap(depth)[:, :, :3]  # Get rid of alpha
                # And back to a tensor
                depth = torch.tensor(depth, dtype=torch.float32)

            # Finally, make batch of 1...
            depth = depth.unsqueeze(0)

            depth_maps.append(depth)

        dav2_model.to(offload_device)

        return (torch.cat(depth_maps),)


NODE_CLASS_MAPPINGS = {
    "DepthAnythingV2Model": DepthAnythingV2Model,
    "DepthAnythingV2": DepthAnythingV2Node,
}
