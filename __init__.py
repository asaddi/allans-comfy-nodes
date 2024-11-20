# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
from .nodes import NODE_CLASS_MAPPINGS

try:
    from . import wdv3_nodes

    NODE_CLASS_MAPPINGS.update(wdv3_nodes.NODE_CLASS_MAPPINGS)
except ImportError:
    pass

try:
    from . import lpips_nodes

    NODE_CLASS_MAPPINGS.update(lpips_nodes.NODE_CLASS_MAPPINGS)
except ImportError:
    pass

try:
    from . import experimental

    NODE_CLASS_MAPPINGS.update(experimental.NODE_CLASS_MAPPINGS)
except ImportError:
    pass

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
