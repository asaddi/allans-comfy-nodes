from enum import Enum, auto
import math
import random


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


class WildText:
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

    TITLE = "Text Wildcards"

    RETURN_TYPES = ("STRING",)

    FUNCTION = "execute"

    CATEGORY = "asaddi"

    def execute(self, text: str, seed: int):
        parsed = parse_wildcards(text)

        rand = random.Random(seed)
        output = []
        for inst, value in parsed:
            assert inst in (
                WildParserState.LITERAL,
                WildParserState.CHOICES,
            ), "bad instruction"
            if inst == WildParserState.LITERAL:
                output.append(value)
            else:
                output.append(rand.choice(value))

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
                    "INT",
                    {
                        "min": 1,
                        "default": 64,
                    },
                ),
            }
        }

    TITLE = "Resolution Chooser"

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "calculate"

    CATEGORY = "asaddi"

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

    def calculate(self, ratio: str, orientation: str, megapixels: float, divisor: int):
        ratio_parts = ratio.split(":", 1)
        ratio_arg = (int(ratio_parts[0]), int(ratio_parts[1]))

        orientation_arg = Orientation[orientation.upper()]

        megapixels_arg = 1024 * 1024 * megapixels

        width, height = ResolutionChooser.find_resolution(
            ratio_arg, orientation_arg, megapixels_arg, divisor
        )

        return (width, height)


NODE_CLASS_MAPPINGS = {
    "WildText": WildText,
    "ResolutionChooser": ResolutionChooser,
}
