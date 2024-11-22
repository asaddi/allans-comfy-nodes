from pathlib import Path
import torch

from .depth_anything_v2.dpt import DepthAnythingV2
from .models import *

from comfy.model_management import get_torch_device, unet_offload_device


MODELS = ModelManager(
    models_subdir="depth_anything_v2",
    allowed_files=["*.pth"],
    config_file="depth_anything_v2.override.yaml",
    default_config_file="depth_anything_v2.yaml",
)
MODELS.load()


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
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
        model.load_state_dict(torch.load(model_path / f'depth_anything_v2_{model_name}.pth', weights_only=True, map_location='cpu'))
        model.eval()

        return (model,)


class DepthAnythingV2Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dav2_model": ("DAV2MODEL",),
                "image": ("IMAGE",),
            },
        }

    TITLE = "DepthAnythingV2"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "private/image"

    def run(self, dav2_model, image):
        torch_device = get_torch_device()
        offload_device = unet_offload_device()

        dav2_model.to(torch_device)

        img = (255. * image).clamp(0, 255).to(torch.uint8)
        # Select first image in batch (batching TODO)
        # And switch from RGB to BGR
        img = img[0, :, :, [2, 1, 0]]

        depth = dav2_model.infer_image(img.numpy())

        dav2_model.to(offload_device)

        depth = torch.tensor(depth, dtype=torch.float32)
        # Rescale to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        # From [H,W] to [H,W,C] where C=1
        depth = depth.unsqueeze(-1)
        # Repeat channel for grayscale
        # TODO Demo app uses matplotlib's Spectral_r colormap, do we want that instead?
        depth = depth.repeat(1, 1, 3)
        # Finally, make batch of 1...
        depth = depth.unsqueeze(0)

        return (depth,)


NODE_CLASS_MAPPINGS = {
    "DepthAnythingV2Model": DepthAnythingV2Model,
    "DepthAnythingV2": DepthAnythingV2Node,
}
