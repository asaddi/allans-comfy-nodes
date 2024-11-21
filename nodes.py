from enum import Enum, auto
import math
from random import Random

from comfy_execution.graph import ExecutionBlocker


# A clean-room low-rent copy of rgthree's seed node.
class PrivateSeed:
    SEED_MIN = 0
    # Note: This is 1 less than what it should be because otherwise on
    # JavaScript side, we cannot generate a random number using
    # Math.random() that is inclusive of the max
    SEED_MAX = 2**53 - 2  # JavaScript's Number.MAX_SAFE_INTEGER-1

    _random = Random()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Avoid calling it "seed" or "noise_seed" so we don't get all
                # the extra widgets.
                "seed_value": (
                    "INT",
                    {
                        "min": -1,
                        "max": cls.SEED_MAX,
                        "default": -1,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def IS_CHANGED(cls, seed_value, **kwargs):
        if seed_value == -1:
            # It will be a random value, make sure we execute
            return float("nan")
        else:
            # Note: There's a minor inefficiency because the
            # frontend can change the seed w/o our knowledge.
            # It just means we execute needlessly once.
            return seed_value

    TITLE = "Seed (private)"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)

    FUNCTION = "run"

    CATEGORY = "private"

    def run(self, seed_value, unique_id, extra_pnginfo):
        if seed_value == -1:
            # Generate seed at random
            seed_value = self._random.randint(self.SEED_MIN, self.SEED_MAX)

        # print(f"seed_value (#{unique_id}) = {seed_value}")

        # store this value in extra_pnginfo, so it gets serialized in the
        # metadata
        unique_id = int(unique_id)
        my_info = [
            node_info
            for node_info in extra_pnginfo["workflow"]["nodes"]
            if node_info["id"] == unique_id
        ][0]
        my_info["widgets_values"] = [seed_value]

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

    @staticmethod
    def get_node_info(extra_pnginfo, node_id) -> dict:
        node_id = int(node_id)
        for node_info in extra_pnginfo["workflow"]["nodes"]:
            if node_info["id"] == node_id:
                return node_info
        # WTF is this?
        raise RuntimeError(f"Node missing from workflow: {node_id}")

    @staticmethod
    def is_output_connected(node_info, out_type) -> bool:
        outputs = node_info.get("outputs", [])
        return any(
            [
                True
                for output in outputs
                if output["type"] == out_type
                and output["links"]  # can be empty list as well
            ]
        )

    @staticmethod
    def is_input_connected(node_info, in_type) -> bool:
        inputs = node_info.get("inputs", [])
        return any(
            [
                True
                for input in inputs
                if input["type"] == in_type and input["link"] is not None
            ]
        )

    @staticmethod
    def get_downstream_nodes(extra_pnginfo, node_id, info_cache):
        downstream = []

        to_check = [int(node_id)]
        while to_check:
            check_id = to_check.pop(0)

            # Yes, consider the starting node downstream from itself as well
            downstream.append(check_id)

            if (node_info := info_cache.get(check_id)) is None:
                node_info = info_cache[check_id] = SimpleBus.get_node_info(
                    extra_pnginfo, check_id
                )

            outputs = node_info.get("outputs", [])
            bus_out = [out["links"] for out in outputs if out["type"] == "SIMPLEBUS"][0]
            if bus_out:
                # NB There can be multiple outputs
                out_ids = [
                    link[3]
                    for link in extra_pnginfo["workflow"]["links"]
                    if link[0] in bus_out
                ]
                to_check.extend(out_ids)

        return downstream

    def _check_downstream_for_type(
        self, downstream, info_cache, type_name, input_value
    ) -> bool:
        if (
            SimpleBus.is_input_connected(info_cache[downstream[0]], type_name)
            and input_value is None
        ):
            for check_id in downstream:
                check_info = info_cache[check_id]
                if SimpleBus.is_output_connected(check_info, type_name):
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
        info_cache = {}
        downstream = SimpleBus.get_downstream_nodes(
            extra_pnginfo, unique_id, info_cache
        )

        # Note: bus is not lazy (but it is optional)

        # The rest of the inputs actually depend on what's being used
        # downstream (used = output is connected to something)

        needed = []

        if self._check_downstream_for_type(downstream, info_cache, "MODEL", model):
            needed.append("model")

        if self._check_downstream_for_type(downstream, info_cache, "VAE", vae):
            needed.append("vae")

        if self._check_downstream_for_type(downstream, info_cache, "LATENT", latent):
            needed.append("latent")

        if self._check_downstream_for_type(downstream, info_cache, "GUIDER", guider):
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

        return (
            bus,
            bus.get("model", ExecutionBlocker("MODEL not present on bus")),
            bus.get("vae", ExecutionBlocker("VAE not present on bus")),
            bus.get("latent", ExecutionBlocker("LATENT not present on bus")),
            bus.get("guider", ExecutionBlocker("GUIDER not present on bus")),
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

        return (width, height)


NODE_CLASS_MAPPINGS = {
    "PrivateSeed": PrivateSeed,
    "SimpleBus": SimpleBus,
    "ReproducibleWildcards": ReproducibleWildcards,
    "ResolutionChooser": ResolutionChooser,
}
