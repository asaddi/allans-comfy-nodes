import csv
import os
from pathlib import Path

import folder_paths


class SaveTabular:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("TABULAR",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                    },
                ),
                "delimiter": (
                    "BOOLEAN",
                    {
                        "label_on": "tab",
                        "label_off": "comma",
                        "default": True,
                    },
                ),
            },
        }

    TITLE = "Save Tabular Data"

    # TODO This is the easiest route, but it means each file will only contain
    # data from a single prompt.
    # Might need some way to accumulate rows into a single file?
    INPUT_IS_LIST = True

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "save_tabular"

    CATEGORY = "private/string"

    def save_tabular(
        self, data: list[list[str]], filename_prefix: list[str], delimiter: list[bool]
    ):
        filename_prefix: str = filename_prefix[0]
        delimiter: bool = delimiter[0]

        # We'll borrow the same method SaveImage uses
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )

        ext = "tsv" if delimiter else "csv"
        file = f"{filename}_{counter:05}_.{ext}"
        with open(os.path.join(full_output_folder, file), "w", newline="") as out:
            # We'll try excel-tab for now, otherwise we might have to come
            # up with our own dialect.
            csvwriter = csv.writer(out, dialect="excel-tab" if delimiter else "excel")
            for row in data:
                csvwriter.writerow(row)

        return ()


class TabularJoin:
    NUM_INPUTS = 3

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "required": {},
            "optional": {},
        }

        for index in range(cls.NUM_INPUTS):
            d["required" if index == 0 else "optional"][f"value{index}"] = (
                "BOOLEAN,FLOAT,INT,STRING,TABULAR",
            )

        return d

    TITLE = "Join Tabular Data"

    RETURN_TYPES = ("TABULAR",)

    FUNCTION = "tab_join"

    CATEGORY = "private/string"

    def tab_join(self, **kwargs):
        result = []
        for index in range(self.NUM_INPUTS):
            value = kwargs.get(f"value{index}")
            if value is not None:
                if isinstance(value, list):
                    result.extend(value)
                else:
                    result.append(str(value))
        return (result,)


class PathSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
            },
        }

    TITLE = "Path Split"

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("filename", "dirname", "basename", "ext")

    FUNCTION = "split_path"

    CATEGORY = "private/string"

    def split_path(self, path: str):
        p = Path(path)

        return (p.name, str(p.parent), p.stem, p.suffix)


class PathJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dirname": ("STRING",),
                "basename": ("STRING",),
            },
            "optional": {
                "ext": ("STRING",),
            },
        }

    TITLE = "Path Join"

    RETURN_TYPES = ("STRING",)
    RETURN_NAME = ("path",)

    FUNCTION = "path_join"

    CATEGORY = "private/string"

    def path_join(self, dirname: str, basename: str, ext: str):
        if not dirname:
            raise ValueError("dirname required")
        if not basename:
            raise ValueError("basename required")

        p = Path(dirname)

        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            basename = basename + ext

        return (str(p / basename),)


NODE_CLASS_MAPPINGS = {
    "SaveTabular": SaveTabular,
    "TabularJoin": TabularJoin,
    "PathSplit": PathSplit,
    "PathJoin": PathJoin,
}
