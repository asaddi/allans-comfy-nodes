from pathlib import Path


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

    CATEGORY = "private/path"

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

    CATEGORY = "private/path"

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
    "PathSplit": PathSplit,
    "PathJoin": PathJoin,
}
