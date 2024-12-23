from pathlib import Path


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
                "TABULAR,INT,FLOAT,STRING",
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
    "TabularJoin": TabularJoin,
    "PathSplit": PathSplit,
    "PathJoin": PathJoin,
}
