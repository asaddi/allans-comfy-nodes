from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import numpy as np

# Do we really need pandas just to parse the CSV?!?!
import pandas as pd
from PIL import Image
import timm
from timm.data import create_transform, resolve_data_config
import torch
from torch import Tensor, nn
import torchvision.transforms.functional

from comfy.model_management import get_torch_device, unet_offload_device


# Much of this generously lifted from https://github.com/neggles/wdv3-timm @ 2f49e85e
# No license?!

MODEL_REPO_MAP = OrderedDict(
    vit="SmilingWolf/wd-vit-tagger-v3",
    swinv2="SmilingWolf/wd-swinv2-tagger-v3",
    convnext="SmilingWolf/wd-convnext-tagger-v3",
)


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(
            f"selected_tags.csv failed to download from {repo_id}"
        ) from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(
        sorted(gen_labels.items(), key=lambda item: item[1], reverse=True)
    )

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(
        sorted(char_labels.items(), key=lambda item: item[1], reverse=True)
    )

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


# FIXME Now I feel dumb for wasting so much time on pad_image
def pad_square(image: Tensor) -> Tensor:
    # Reminder: ComfyUI images are [B,H,W,C]
    _, height, width, _ = image.shape

    # determine largest dimension
    px = max(height, width)

    # create new tensor with px X px image
    # FIXME I feel like "1.0" is a major assumption. Also, is it always float32?
    channel_max = 1.0 if image.dtype.is_floating_point else torch.iinfo(image.dtype).max
    canvas = torch.full(
        (image.shape[0], px, px, image.shape[3]),
        channel_max,
        dtype=image.dtype,
        device=image.device,
    )

    # calculate top-left corner
    y = (px - height) // 2
    x = (px - width) // 2

    canvas[:, y : (y + height), x : (x + width), :] = image

    return canvas


def format_tags(tags) -> str:
    return (
        ", ".join([x for x in tags])
        .replace("_", " ")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


@dataclass
class RatingDisplay:
    label: str
    value: float

    @property
    def formatted(self):
        return f"{self.label}:{self.value:.3f}"


class WDv3Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([k for k in MODEL_REPO_MAP.keys()],),
            }
        }

    TITLE = "WDv3 Model"

    RETURN_TYPES = ("WDV3_MODEL",)

    FUNCTION = "execute"

    CATEGORY = "private/wdv3"

    def execute(self, model):
        repo_id = MODEL_REPO_MAP[model]

        print(f"Loading model '{model}' from '{repo_id}'...")
        model: nn.Module = timm.create_model("hf-hub:" + repo_id).eval()
        state_dict = timm.models.load_state_dict_from_hf(repo_id)
        model.load_state_dict(state_dict)

        print("Loading tag list...")
        labels: LabelData = load_labels_hf(repo_id=repo_id)

        print("Creating data transform...")
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        return ({"model": model, "labels": labels, "transform": transform},)


class WDv3Tagger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wdv3_model": ("WDV3_MODEL",),
                "image": ("IMAGE",),
                "gen_threshold": (
                    "FLOAT",
                    {
                        "min": 0.0,
                        "max": 1.0,
                        "default": 0.35,
                    },
                ),
                "char_threshold": (
                    "FLOAT",
                    {
                        "min": 0.0,
                        "max": 1.0,
                        "default": 0.75,
                    },
                ),
            }
        }

    TITLE = "WDv3 Tagger"

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("general_tags", "char_tags", "rating_tags")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "execute"

    CATEGORY = "private/wdv3"

    def execute(
        self, wdv3_model, image: Tensor, gen_threshold: float, char_threshold: float
    ):
        labels = wdv3_model["labels"]
        transform = wdv3_model["transform"]
        model = wdv3_model["model"]

        torch_device = get_torch_device()
        offload_device = unet_offload_device()

        # move model to GPU, if available
        model = model.to(torch_device)

        image = image.permute(0, 3, 1, 2)  # To [B,C,H,W], suitable for to_pil_image

        # The output lists
        general_tags = []
        char_tags = []
        rating_tags = []

        # TODO How to deal with batched images properly?
        for img in image:
            print("Loading image and preprocessing...")
            # This seems kinda dumb, but I'm going to treat transform like a
            # blackbox and only feed it PIL images.
            img_input = torchvision.transforms.functional.to_pil_image(img)
            # ensure image is RGB
            img_input = pil_ensure_rgb(img_input)
            # pad to square with white background
            img_input = pil_pad_square(img_input)
            # run the model's input transform to convert to tensor and rescale
            inputs: Tensor = transform(img_input).unsqueeze(0)
            # NCHW image RGB to BGR
            inputs = inputs[:, [2, 1, 0]]

            print("Running inference...")
            with torch.inference_mode():
                inputs = inputs.to(torch_device)
                # run the model
                outputs = model.forward(inputs)
                # apply the final activation function (timm doesn't support doing this internally)
                outputs = nn.functional.sigmoid(outputs)
                # move output to CPU
                outputs = outputs.to("cpu")

            print("Processing results...")
            caption, taglist, ratings, character, general = get_tags(
                probs=outputs.squeeze(0),
                labels=labels,
                gen_threshold=gen_threshold,
                char_threshold=char_threshold,
            )

            general_tags.append(format_tags(general))
            char_tags.append(format_tags(character))

            # The rating tags are static making their probability/confidence
            # level significant.
            # I'm not sure of a good textual format for them yet, so I'll
            # just go with this for now. See RatingDisplay.formatted
            output_ratings = [
                RatingDisplay(label=label, value=value)
                for label, value in ratings.items()
            ]
            output_ratings.sort(key=lambda x: x.value, reverse=True)

            rating_tags.append(", ".join([r.formatted for r in output_ratings]))

        # move model to offload device
        model = model.to(offload_device)

        return (general_tags, char_tags, rating_tags)


NODE_CLASS_MAPPINGS = {
    "WDv3Model": WDv3Model,
    "WDv3Tagger": WDv3Tagger,
}
