# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
from importlib import import_module

from .nodes import NODE_CLASS_MAPPINGS


modules = [
    "wdv3_nodes",
    "lpips_nodes",
    "dav2_nodes",
    "experimental",
]

for mod_name in modules:
    try:
        mod = import_module(f".{mod_name}", package=__name__)
    except ImportError:
        continue

    NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)


NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
