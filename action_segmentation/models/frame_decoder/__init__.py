from action_segmentation.utils.d2_compat import Registry


from .build import build_frame_decoder, FRAME_DECODER_REGISTRY
from .asformer_encoder import ASFormerEncoder
from .sstcn_encoder import SSTCNEncoder
