"""Compatibility shim for the small subset of detectron2 used by this repo.

Only three symbols are needed: ``configurable`` (decorator that routes
construction through ``from_config`` when a CfgNode is passed), ``Registry``
(name -> class registry), and ``get_world_size``. The real detectron2 is used
when available; otherwise we fall back to in-repo / fvcore equivalents so the
project is runnable without compiling detectron2.
"""

try:
    from action_segmentation.utils.d2_compat import configurable
    from action_segmentation.utils.d2_compat import Registry
    from detectron2.utils.comm import get_world_size
except Exception:
    from fvcore.common.registry import Registry
    from action_segmentation.config.config_node import configurable
    from action_segmentation.utils.dist import get_world_size

__all__ = ["configurable", "Registry", "get_world_size"]
