# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
from .nodes import NODE_CLASS_MAPPINGS

if True:
    from . import wdv3

    NODE_CLASS_MAPPINGS.update(wdv3.NODE_CLASS_MAPPINGS)

try:
    import lpips
    from . import lpips_node

    NODE_CLASS_MAPPINGS.update(lpips_node.NODE_CLASS_MAPPINGS)
except ImportError:
    pass

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
