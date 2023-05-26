from typing import Dict, List, Optional, Tuple, Union
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.modeling.proposal_generator import RPN
from detectron2.layers import nonzero_tuple

import torch
import torch.nn.functional as F
from torch import nn

@PROPOSAL_GENERATOR_REGISTRY.register()
class IOURPN(RPN):

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_ious,

    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        gt_ious = torch.stack(gt_ious)
        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        if self.box_reg_loss_type == "smooth_l1":
            anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)
            gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
            gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)
            localization_loss = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            pred_proposals = cat(pred_proposals, dim=1)
            pred_proposals = pred_proposals.view(-1, pred_proposals.shape[-1])
            pos_mask = pos_mask.view(-1)
            localization_loss = giou_loss(
                pred_proposals[pos_mask], cat(gt_boxes)[pos_mask], reduction="sum"
            )
        else:
            raise ValueError(f"Invalid rpn box reg loss type '{self.box_reg_loss_type}'")

        valid_mask = gt_labels >= 0
        # objectness_loss = F.binary_cross_entropy_with_logits(
        #     cat(pred_objectness_logits, dim=1)[valid_mask],
        #     gt_labels[valid_mask].to(torch.float32),
        #     reduction="sum",
        # )
        # use sigmoid func
        pred_objectness_logits = [logits.sigmoid() for logits in pred_objectness_logits]
        iouness_loss = smooth_l1_loss(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_ious[valid_mask].to(torch.float32),
            self.smooth_l1_beta,
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": iouness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def forward(self,
                images,
                features,
                gt_instances):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_iouness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_iouness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_iouness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes, gt_ious = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_iouness_logits, gt_labels, pred_anchor_deltas, gt_boxes, gt_ious
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_iouness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        matched_ious = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            match_quality_matrix = match_quality_matrix.permute(1, 0)
            num_anchors = match_quality_matrix.size(0)

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
                matched_ious_i = match_quality_matrix[range(num_anchors), matched_idxs]

            del match_quality_matrix
            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_ious.append(matched_ious_i)
        return gt_labels, matched_gt_boxes, matched_ious
    
#     def _subsample_labels(self, label):
#         """
#         Randomly sample a subset of positive and negative examples, and overwrite
#         the label vector to the ignore value (-1) for all elements that are not
#         included in the sample.

#         Args:
#             labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
#         """
#         pos_idx, neg_idx = subsample_labels(
#             label, self.batch_size_per_image, self.positive_fraction, 0
#         )
#         # Fill with the ignore label (-1), then set positive and negative labels
#         label.fill_(-1)
#         label.scatter_(0, pos_idx, 1)
#         label.scatter_(0, neg_idx, 0)
#         return label

# def subsample_labels(
#     labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
# ):
#     """
#     Return `num_samples` (or fewer, if not enough found)
#     random samples from `labels` which is a mixture of positives & negatives.
#     It will try to return as many positives as possible without
#     exceeding `positive_fraction * num_samples`, and then try to
#     fill the remaining slots with negatives.

#     Args:
#         labels (Tensor): (N, ) label vector with values:
#             * -1: ignore
#             * bg_label: background ("negative") class
#             * otherwise: one or more foreground ("positive") classes
#         num_samples (int): The total number of labels with value >= 0 to return.
#             Values that are not sampled will be filled with -1 (ignore).
#         positive_fraction (float): The number of subsampled labels with values > 0
#             is `min(num_positives, int(positive_fraction * num_samples))`. The number
#             of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
#             In order words, if there are not enough positives, the sample is filled with
#             negatives. If there are also not enough negatives, then as many elements are
#             sampled as is possible.
#         bg_label (int): label index of background ("negative") class.

#     Returns:
#         pos_idx, neg_idx (Tensor):
#             1D vector of indices. The total length of both is `num_samples` or fewer.
#     """
#     positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
#     negative = nonzero_tuple(labels == bg_label)[0]

#     num_pos = min(num_samples, positive.numel())
#     num_neg = num_samples - num_pos
#     num_neg = min(num_neg, negative.numel())

#     # randomly select positive and negative examples
#     perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
#     perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

#     pos_idx = positive[perm1]
#     neg_idx = negative[perm2]
#     return pos_idx, neg_idx
