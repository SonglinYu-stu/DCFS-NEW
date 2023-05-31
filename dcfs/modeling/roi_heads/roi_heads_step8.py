import torch
import logging
import numpy as np
from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
# from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from .mask_head import build_mask_head
from .roi_heads import Res5ROIHeads, select_foreground_proposals
# from ..meta_arch.adapter import AdapterLayer

from .roi_heads import ROI_HEADS_REGISTRY
import torch.nn.init as init
from detectron2.layers import nonzero_tuple
from .classes import COCO_BASE_CLASSES, COCO_ALL_CLASSES, COCO_NOVEL_CLASSES
import json
from .fast_rcnn import FastRCNNOutputsStep8
import torch.nn.functional as F
from detectron2.utils.events import get_event_storage

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsStep8(Res5ROIHeads):

    def __init__(self,
                 *args,
                 clip_feat='./dcfs/modeling/roi_heads/coco_clip.json',
                 scale=20,
                **kwargs):
        super().__init__(*args, **kwargs)
        cfg = args[0]
        self.base_train = cfg.MODEL.ROI_HEADS.BASETRAIN
        self.base_classes = COCO_BASE_CLASSES
        self.novel_classes = COCO_NOVEL_CLASSES
        self.all_classes = COCO_ALL_CLASSES
        self.pot_thresh = cfg.MODEL.ROI_HEADS.POT_THRESH # potential thresh hold
        # self.novel_thresh = [0.0] * len(COCO_NOVEL_CLASSES) # each novel cls thresh hold
        with open(clip_feat, 'r') as f:
            self.clip_feat = json.load(f)
        basecls_feat = []
        novelcls_feat = []
        for cls in self.base_classes:
            basecls_feat.append(torch.tensor(self.clip_feat[cls]))
        for cls in self.novel_classes:
            novelcls_feat.append(torch.tensor(self.clip_feat[cls]))
        self.baseclip_feat = basecls_feat
        self.novelclip_feat = novelcls_feat
        self.scale = nn.Parameter(torch.ones(1) * scale)

    def cosine_similarity(self, tensor1, tensor2):
        '''
        tensor1: [n, c]
        tensor2: [m, c]
        '''
        # 归一化两个张量
        tensor1_normalized = F.normalize(tensor1, p=2, dim=1)
        tensor2_normalized = F.normalize(tensor2, p=2, dim=1)

        # 计算余弦相似度
        similarity = torch.mm(tensor1_normalized, tensor2_normalized.t())
        return similarity

    def log(self, name, value):
        storage = get_event_storage()
        storage.put_scalar(
           name, value
        )
        

    @torch.no_grad()
    def find_potential(self, proposals, proposal_embeddings, 
                       baseclip_feat, novelclip_feat):

        num_base = len(self.base_classes)
        bg_class_id = num_base
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_id)
        bg_inds = (gt_classes == bg_class_id)
        
        # update self.pot_thresh
        fg_gt_cls = gt_classes[fg_inds]
        fg_clsfeat = proposal_embeddings[fg_inds]
        
        # import pdb
        # pdb.set_trace()
        fg_cossim = self.cosine_similarity(fg_clsfeat, baseclip_feat) # [k, 60]
        fg_cossim = fg_cossim[range(fg_cossim.size(0)), fg_gt_cls] # [k]
        fg_cossim_mean = torch.mean(fg_cossim)
        fg_cossim_std = torch.std(fg_cossim)
        # 判nan
        fg_cossim_mean[torch.isnan(fg_cossim_mean)] = .0
        fg_cossim_std[torch.isnan(fg_cossim_mean)] = .0
        self.log('fg_cossim_mean', fg_cossim_mean)
        self.log('fg_cossim_std', fg_cossim_std)

        bg_clsfeat = proposal_embeddings[bg_inds]
        bg_cossim = self.cosine_similarity(bg_clsfeat, novelclip_feat) # [k, 20]
        bg_cossim_max, pot_cls_ids = bg_cossim.max(dim=1)
        if bg_cossim_max.size(0):    
            self.log('bg_cossim_max', bg_cossim_max.max())
        else:
            self.log('bg_cossim_max', torch.zeros(1).to(bg_cossim_max))
        pot_thresh = fg_cossim_mean - fg_cossim_std
        pot_thresh = torch.max(pot_thresh, torch.tensor(self.pot_thresh).to(pot_thresh))
        pot_flags = bg_cossim_max > pot_thresh

        neg_classes = gt_classes[bg_inds]
        neg_classes[pot_flags] = pot_cls_ids[pot_flags] + num_base + 1
        gt_classes[bg_inds] = neg_classes

        return gt_classes, pot_cls_ids, pot_flags

    def forward(self, images, features, proposals, targets=None, fs_class=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas, proposal_embeddings = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        if self.training and self.base_train:
            baseclip_feat = torch.stack(self.baseclip_feat).to(proposal_embeddings)
            novelclip_feat = torch.stack(self.novelclip_feat).to(proposal_embeddings)
            allclip_feat = torch.cat([novelclip_feat, baseclip_feat])
            origin_gt_classes = torch.cat([p.gt_classes for p in proposals])
            gt_classes, pot_cls_ids, pot_flags = self.find_potential(proposals,
                                                                    proposal_embeddings,
                                                                    baseclip_feat,
                                                                    novelclip_feat)
            bg_inds = (origin_gt_classes == len(self.base_classes))
            bg_clsfeat = proposal_embeddings[bg_inds]
            pot_clsfeat = bg_clsfeat[pot_flags]

            pot_base_scores = self.scale * self.cosine_similarity(pot_clsfeat, baseclip_feat) # [k, 60]
            pot_novel_scores = self.scale * self.cosine_similarity(pot_clsfeat, novelclip_feat) # [k, 20]

            pot_base_scores_max, _ = pot_base_scores.max(dim=1) # [k]
            pot_novel_scores_max, _ = pot_novel_scores.max(dim=1) # [k]

        outputs = FastRCNNOutputsStep8(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_class_loss,
            fs_class
        )

        if self.training:
            del features
            if self.base_train:
                losses = outputs.baseloss(gt_classes, pot_base_scores_max, pot_novel_scores_max)
            else:
                losses = outputs.losses()

            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}