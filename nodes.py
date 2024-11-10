from enum import Enum, auto
import math


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
    "ResolutionChooser": ResolutionChooser,
}
