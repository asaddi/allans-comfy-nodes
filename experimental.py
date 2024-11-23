from pprint import pprint

import torch


class DumpToConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_pprint": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "pprint",
                        "label_off": "raw",
                    },
                ),
                "input": ("*",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    TITLE = "Dump To Console"

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "run"

    CATEGORY = "private/debug"

    def run(self, use_pprint, input, unique_id):
        print(f"DumpToConsole (#{unique_id}) = ", end="")
        if use_pprint:
            pprint(input)
        else:
            print(input)

        return ()


class CLIPDistance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip1": ("CONDITIONING",),
                "clip2": ("CONDITIONING",),
            },
        }

    TITLE = "CLIP Distance"

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "calculate"

    CATEGORY = "private/experimental"

    def calculate(self, clip1, clip2):
        clip1: torch.Tensor = clip1[0][0]
        clip2: torch.Tensor = clip2[0][0]

        print(clip1)
        print(clip2)
        dist = torch.dist(clip1, clip2)
        print(f"CLIP distance = {dist}")

        return ()


NODE_CLASS_MAPPINGS = {
    "DumpToConsole": DumpToConsole,
    "CLIPDistance": CLIPDistance,
}
