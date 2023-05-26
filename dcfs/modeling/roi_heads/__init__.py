from .box_head import ROI_BOX_HEAD_REGISTRY, build_box_head
from .roi_heads import (
    ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, build_roi_heads, select_foreground_proposals)
# from .roi_heads_step7 import Res5ROIHeadsStep7
from .roi_heads_step8 import Res5ROIHeadsStep8