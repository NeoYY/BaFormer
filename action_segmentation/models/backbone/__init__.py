from action_segmentation.utils.d2_compat import Registry


from .build import build_backbone, BACKBONE_REGISTRY
from .mstcn_model import SSTCN
from .asformer_model import ASFormerEncoder
