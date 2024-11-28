from enum import Enum, auto
import math

import torch

from .utils import WorkflowUtils

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder
import folder_paths


class EmptyLatentImageSelector:
    # Note: This is from ComfyUI's nodes.py, but I don't want to import it.
    MAX_RESOLUTION = 16384

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # NB EmptySD3LatentImage has step 16
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": cls.MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "The width of the latent images in pixels.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": cls.MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "The height of the latent images in pixels.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "The number of latent images in the batch.",
                    },
                ),
                "type": (
                    ["sd1", "sdxl", "sd3", "flux"],
                    {
                        "default": "sdxl",
                    },
                ),
            }
        }

    TITLE = "Empty Latent Image Selector"
    DESCRIPTION = (
        "Create a new batch of empty latent images to be denoised via sampling."
    )

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)

    FUNCTION = "generate"

    CATEGORY = "private/latent"

    def generate(self, width, height, batch_size, type):
        # It's just torch.zeros() of shape [B,C,H,W] with C=4 for sd1/sdxl
        # and C=16 for sd3/flux.
        # Implement it myself or do it via node expansion?

        # Node expansion is probably more future-proof and much easier
        # to implement other, future, latents...
        # Plus there's the caching advantage, but I don't think it matters
        # here...
        graph = GraphBuilder()

        if type in ("sd1", "sdxl"):
            # Use EmptyLatentImage
            width = (width // 8) * 8
            height = (height // 8) * 8
            empty_latent = graph.node(
                "EmptyLatentImage", width=width, height=height, batch_size=batch_size
            )
        else:
            # Use EmptySD3LatentImage
            width = (width // 16) * 16
            height = (height // 16) * 16
            empty_latent = graph.node(
                "EmptySD3LatentImage", width=width, height=height, batch_size=batch_size
            )

        return {
            "result": (empty_latent.out(0),),
            "expand": graph.finalize(),
        }


class ImageDimensions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    TITLE = "Get Image Dimensions"

    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "run"

    CATEGORY = "private/image"

    def run(self, image: torch.Tensor):
        height, width = image.shape[1:3]
        return {"ui": {"dims": ([width, height],)}, "result": (width, height)}


class PrivateLoraStack:
    LORA_STACK_COUNT = 4

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }
        }

        for index in range(cls.LORA_STACK_COUNT):
            d["required"][f"lora_name_{index}"] = (
                ["None"] + folder_paths.get_filename_list("loras"),
            )
            d["required"][f"strength_{index}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                },
            )

        return d

    TITLE = "LoRA Stack"

    RETURN_TYPES = ("MODEL", "CLIP")

    FUNCTION = "load_loras"

    CATEGORY = "private/loaders"

    def load_loras(self, model, clip, **kwargs):
        graph = GraphBuilder()
        stack = []
        for index in range(self.LORA_STACK_COUNT):
            lora_name = kwargs[f"lora_name_{index}"]
            strength = kwargs[f"strength_{index}"]

            if lora_name == "None" or strength == 0.0:
                continue

            node = graph.node(
                "LoraLoader",
                model=model,
                clip=clip,
                lora_name=lora_name,
                strength_model=strength,
                strength_clip=strength,
            )
            stack.append(node)

            model = node.out(0)
            clip = node.out(1)

        return {
            "result": (model, clip),
            "expand": graph.finalize(),
        }


class StyleModelApplyStrength:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "strength": (
                    "FLOAT",
                    {
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "default": 1.0,
                    },
                ),
            }
        }

    TITLE = "Apply Style Model with Strength"

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "expand"

    CATEGORY = "private/conditioning"

    def expand(self, conditioning, style_model, clip_vision_output, strength):
        graph = GraphBuilder()
        apply_style_model_node = graph.node(
            "StyleModelApply",
            conditioning=conditioning,
            style_model=style_model,
            clip_vision_output=clip_vision_output,
        )
        conditioning_average_node = graph.node(
            "ConditioningAverage",
            conditioning_to=apply_style_model_node.out(0),
            conditioning_from=conditioning,
            conditioning_to_strength=strength,
        )
        return {
            "result": (conditioning_average_node.out(0),),
            "expand": graph.finalize(),
        }


# A clean-room low-rent copy of rgthree's seed node.
class PrivateSeed:
    SEED_MIN = 0
    # Note: This is 1 less than what it should be because otherwise on
    # JavaScript side, we cannot generate a random number using
    # Math.random() that is inclusive of the max
    SEED_MAX = 2**53 - 2  # JavaScript's Number.MAX_SAFE_INTEGER-1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Avoid calling it "seed" or "noise_seed" so we don't get all
                # the extra widgets.
                "seed_value": (
                    "INT",
                    {
                        "min": cls.SEED_MIN,
                        "max": cls.SEED_MAX,
                        "default": cls.SEED_MIN,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def IS_CHANGED(cls, seed_value, unique_id, extra_pnginfo):
        return seed_value

    TITLE = "Seed"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)

    FUNCTION = "run"

    CATEGORY = "private"

    def run(self, seed_value, unique_id, extra_pnginfo):
        # print(f"seed_value (#{unique_id}) = {seed_value}")

        # In the metadata, change this node into a fixed seed
        unique_id = int(unique_id)
        my_info = [
            node_info
            for node_info in extra_pnginfo["workflow"]["nodes"]
            if node_info["id"] == unique_id
        ][0]
        my_info["properties"]["randomizeSeed"] = False

        # pass the value back up to the UI so it can update the button
        return {"ui": {"seed_value": (seed_value,)}, "result": (seed_value,)}


class SimpleBus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "bus": ("SIMPLEBUS",),
                "model": (
                    "MODEL",
                    {
                        "lazy": True,
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "lazy": True,
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "lazy": True,
                    },
                ),
                "guider": (
                    "GUIDER",
                    {
                        "lazy": True,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    TITLE = "Simple Bus"

    RETURN_TYPES = ("SIMPLEBUS", "MODEL", "VAE", "LATENT", "GUIDER")
    RETURN_NAMES = ("BUS", "MODEL", "VAE", "LATENT", "GUIDER")

    FUNCTION = "execute"

    CATEGORY = "private"

    def _check_downstream_for_type(
        self, downstream, wf_utils: WorkflowUtils, type_name, input_value
    ) -> bool:
        if (
            wf_utils.is_input_connected(downstream[0], type=type_name)
            and input_value is None
        ):
            for check_id in downstream:
                if wf_utils.is_output_connected(check_id, type=type_name):
                    return True
        return False

    def _check_downstream_for_name(
        self, downstream, wf_utils: WorkflowUtils, name, input_value
    ) -> bool:
        if (
            wf_utils.is_input_connected(downstream[0], name=name)
            and input_value is None
        ):
            for check_id in downstream:
                if wf_utils.is_output_connected(check_id, name=name):
                    return True
        return False

    def check_lazy_status(
        self,
        unique_id,
        extra_pnginfo,
        bus=None,
        model=None,
        vae=None,
        latent=None,
        guider=None,
    ):
        unique_id = int(unique_id)

        # TODO do all this lazily
        wf_utils = WorkflowUtils(extra_pnginfo)
        downstream = wf_utils.get_downstream_nodes(unique_id, "SIMPLEBUS")

        # Note: bus is not lazy (but it is optional)

        # The rest of the inputs actually depend on what's being used
        # downstream (used = output is connected to something)

        needed = []

        if self._check_downstream_for_type(downstream, wf_utils, "MODEL", model):
            needed.append("model")

        if self._check_downstream_for_type(downstream, wf_utils, "VAE", vae):
            needed.append("vae")

        if self._check_downstream_for_type(downstream, wf_utils, "LATENT", latent):
            needed.append("latent")

        if self._check_downstream_for_type(downstream, wf_utils, "GUIDER", guider):
            needed.append("guider")

        # print(f"SimpleBus #{unique_id} needed: {needed}")
        return needed

    def execute(
        self,
        unique_id,
        extra_pnginfo,
        bus=None,
        model=None,
        vae=None,
        latent=None,
        guider=None,
    ):
        if bus is None:
            bus = {}

        if model is not None:
            bus["model"] = model

        if vae is not None:
            bus["vae"] = vae

        if latent is not None:
            bus["latent"] = latent

        if guider is not None:
            bus["guider"] = guider

        # None of these are tensors (and therefore are boolean ambiguous),
        # but we'll be consistent anyway (with ControlBus)
        model = bus.get("model")
        vae = bus.get("vae")
        latent = bus.get("latent")
        guider = bus.get("guider")

        return (
            bus,
            model
            if model is not None
            else ExecutionBlocker("MODEL not present on bus"),
            vae if vae is not None else ExecutionBlocker("VAE not present on bus"),
            latent
            if latent is not None
            else ExecutionBlocker("LATENT not present on bus"),
            guider
            if guider is not None
            else ExecutionBlocker("GUIDER not present on bus"),
        )


class ControlBus(SimpleBus):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "cbus": ("CONTROLBUS",),
                "positive": (
                    "CONDITIONING",
                    {
                        "lazy": True,
                    },
                ),
                "negative": (
                    "CONDITIONING",
                    {
                        "lazy": True,
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "lazy": True,
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "lazy": True,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    TITLE = "Control Bus"

    RETURN_TYPES = ("CONTROLBUS", "CONDITIONING", "CONDITIONING", "IMAGE", "MASK")
    RETURN_NAMES = ("CBUS", "positive", "negative", "IMAGE", "MASK")

    def check_lazy_status(
        self,
        unique_id,
        extra_pnginfo,
        cbus=None,
        positive=None,
        negative=None,
        image=None,
        mask=None,
    ):
        unique_id = int(unique_id)

        wf_utils = WorkflowUtils(extra_pnginfo)
        downstream = wf_utils.get_downstream_nodes(unique_id, "CONTROLBUS")

        # Note: bus is not lazy (but it is optional)

        # The rest of the inputs actually depend on what's being used
        # downstream (used = output is connected to something)

        needed = []

        if self._check_downstream_for_name(downstream, wf_utils, "positive", positive):
            needed.append("positive")

        if self._check_downstream_for_name(downstream, wf_utils, "negative", negative):
            needed.append("negative")

        if self._check_downstream_for_type(downstream, wf_utils, "IMAGE", image):
            needed.append("image")

        if self._check_downstream_for_type(downstream, wf_utils, "MASK", mask):
            needed.append("mask")

        # print(f"ControlBus #{unique_id} needed: {needed}")
        return needed

    def execute(
        self,
        unique_id,
        extra_pnginfo,
        cbus=None,
        positive=None,
        negative=None,
        image=None,
        mask=None,
        **kwargs,
    ):
        if cbus is None:
            cbus = {}

        if positive is not None:
            cbus["positive"] = positive

        if negative is not None:
            cbus["negative"] = negative

        if image is not None:
            cbus["image"] = image

        if mask is not None:
            cbus["mask"] = mask

        # image & mask are pure tensors and are therefore
        # ambiguous. Just check all of them against None.
        positive = cbus.get("positive")
        negative = cbus.get("negative")
        image = cbus.get("image")
        mask = cbus.get("mask")

        return (
            cbus,
            positive
            if positive is not None
            else ExecutionBlocker("positive not present on bus"),
            negative
            if negative is not None
            else ExecutionBlocker("negative not present on bus"),
            image
            if image is not None
            else ExecutionBlocker("IMAGE not present on bus"),
            mask if mask is not None else ExecutionBlocker("MASK not present on bus"),
        )


class WildParserState(Enum):
    LITERAL = auto()
    ESCAPE = auto()
    CHOICES = auto()
    CHOICES_ESCAPE = auto()


# Who's crazy enough to write a parser by hand in 2024?!
def parse_wildcards(input: str):
    output = []

    state = WildParserState.LITERAL

    lit = []
    choice = []
    choices = []

    ESCAPABLE = ("\\", "{", "|", "}")

    for c in input:
        if state == WildParserState.LITERAL:
            if c == "\\":
                state = WildParserState.ESCAPE
            elif c == "{":
                if len(lit) > 0:
                    output.append((WildParserState.LITERAL, "".join(lit)))
                    lit = []

                choice = []
                choices = []
                state = WildParserState.CHOICES
            else:
                lit.append(c)

        elif state == WildParserState.ESCAPE:
            if c in ESCAPABLE:
                lit.append(c)
            else:
                # Not a valid escape, just forget we were escaped
                lit.append("\\")
                lit.append(c)
            state = WildParserState.LITERAL

        elif state == WildParserState.CHOICES:
            if c == "\\":
                state = WildParserState.CHOICES_ESCAPE
            elif c == "}":
                if len(choice) > 0:
                    choices.append("".join(choice))
                if len(choices) > 0:
                    output.append((WildParserState.CHOICES, choices))
                state = WildParserState.LITERAL
            elif c == "|":
                if len(choice) > 0:
                    choices.append("".join(choice))
                choice = []
            else:
                choice.append(c)

        elif state == WildParserState.CHOICES_ESCAPE:
            if c in ESCAPABLE:
                choice.append(c)
            else:
                # Not a valid escape
                choice.append("\\")
                choice.append(c)
            state = WildParserState.CHOICES

    # Finished processing the entire string.
    # We *should* error out if the state isn't LITERAL, but
    # we actually don't have that luxury. Just clean up and make do.

    if state == WildParserState.LITERAL:
        if len(lit) > 0:
            output.append((WildParserState.LITERAL, "".join(lit)))
    elif state == WildParserState.ESCAPE:
        lit.append("\\")
        output.append((WildParserState.LITERAL, "".join(lit)))
    elif state == WildParserState.CHOICES:
        if len(choice) > 0:
            choices.append("".join(choice))
        if len(choices) > 0:
            output.append((WildParserState.CHOICES, choices))
    elif state == WildParserState.CHOICES_ESCAPE:
        choice.append("\\")
        choices.append("".join(choice))
        # We know it has at least one choice...
        output.append((WildParserState.CHOICES, choices))

    return output


class ReproducibleWildcards:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "min": 0,
                        "max": 0xFFFF_FFFF_FFFF_FFFF,
                    },
                ),
            }
        }

    TITLE = "Reproducible Wildcards"

    RETURN_TYPES = ("STRING",)

    FUNCTION = "execute"

    CATEGORY = "private"

    def execute(self, text: str, seed: int):
        parsed = parse_wildcards(text)

        output = []
        for inst, value in parsed:
            assert inst in (
                WildParserState.LITERAL,
                WildParserState.CHOICES,
            ), "bad instruction"
            if inst == WildParserState.LITERAL:
                output.append(value)
            else:
                count = len(value)
                choice = seed % count
                output.append(value[choice])
                seed = seed // count

        return ("".join(output),)


class Orientation(Enum):
    PORTRAIT = auto()
    LANDSCAPE = auto()


class ResolutionChooser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (
                    [
                        "1:1",
                        "1:2",
                        "2:3",
                        "3:4",
                        "4:5",
                        "16:10",
                    ],
                ),
                "orientation": (
                    [
                        "portrait",
                        "landscape",
                    ],
                ),
                "megapixels": ("FLOAT", {"min": 0.0, "default": 1.0}),
                "divisor": (
                    [
                        "64",  # SDXL/SD3.5
                        "16",  # Flux
                        "8",  # SD1.5
                    ],
                ),
            }
        }

    TITLE = "Resolution Chooser"

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "calculate"

    CATEGORY = "private"

    @staticmethod
    def _round(size: int, step: int, up: bool = True) -> int:
        if up:
            return int(step * ((size + step - 1) // step))
        else:
            return int(step * (size // step))

    @staticmethod
    def find_resolution(
        ratio: tuple[int, int],
        orientation: Orientation,
        pixel_count: int,
        pixel_step: int,
    ):
        # Force portrait
        if ratio[1] < ratio[0]:
            ratio = (ratio[1], ratio[0])

        # ...so width is never longer than height

        scale = pixel_count / (ratio[0] * ratio[1] * pixel_step * pixel_step)
        scale = math.sqrt(scale)

        # round up on longest dimension
        height = ResolutionChooser._round(
            ratio[1] * scale * pixel_step, pixel_step, up=True
        )
        width = pixel_count // height
        # round down on the other
        width = ResolutionChooser._round(width, pixel_step, up=False)

        # Switch to desired orientation
        if orientation == Orientation.LANDSCAPE:
            width, height = height, width

        return width, height

    def calculate(self, ratio: str, orientation: str, megapixels: float, divisor: str):
        ratio_parts = ratio.split(":", 1)
        ratio_arg = (int(ratio_parts[0]), int(ratio_parts[1]))

        orientation_arg = Orientation[orientation.upper()]

        megapixels_arg = 1024 * 1024 * megapixels

        width, height = ResolutionChooser.find_resolution(
            ratio_arg, orientation_arg, megapixels_arg, int(divisor)
        )

        return {"ui": {"dims": ([width, height],)}, "result": (width, height)}


NODE_CLASS_MAPPINGS = {
    "EmptyLatentImageSelector": EmptyLatentImageSelector,
    "ImageDimensions": ImageDimensions,
    "PrivateLoraStack": PrivateLoraStack,
    "StyleModelApplyStrength": StyleModelApplyStrength,
    "PrivateSeed": PrivateSeed,
    "SimpleBus": SimpleBus,
    "ControlBus": ControlBus,
    "ReproducibleWildcards": ReproducibleWildcards,
    "ResolutionChooser": ResolutionChooser,
}
