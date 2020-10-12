import torch
import torch.nn.functional as F
from torch import nn

from pytorch_mask_rcnn.model.box_ops import BoxCoder, box_iou, process_box, nms
from pytorch_mask_rcnn.model.utils import Matcher, BalancedPositiveNegativeSampler

class RPNhead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)

        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        def forward(self, x):
            x = F.relu(self.conv(x))
            logits = self.cls_logits(x)
            bbox_reg = self.bbox_pred(x)
            return logits, bbox_reg

class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nums_thresh):
        super().__init__()

        self.anchor_generator