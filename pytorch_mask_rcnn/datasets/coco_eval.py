import copy
import torch
import numpy as np

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

class CocoEvalutor:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]

        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        #self.ann_labels = ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iou_types=iou_types)
                          for iou_type in iou_types}

    def accumulate(self, coco_results):
        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_dt = self.coco_gt.loadRes(coco_results) if coco_results else COCO()

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate()

    def summarize(self):
        for iou_type in self.iou_types:
            print("IoI metric: {}".format(iou_type))
            self.coco_eval[iou_type].summarize()

def perpare_for_coco(predictions, ann_labels):
    coco_results = []
    for original_id, prediction in predictions.item():
        if len(prediction) == 0:
            continue

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    masks = prediction["masks"]

    x1, y1, x2, y2 = boxes.unbind(1)
    boxes = torch.stack((x1, x2 - x1, y2 - y1), dim=1)
    boxes = boxes.tolist()
    scores = prediction["scores"].tolist()
    labels = prediction["labels"].tolist()
    labels = [ann_labels[l] for l in labels]

    masks = masks > 0.5
    rles = [
        masks_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
        for mask in masks
    ]
    for rle in rles:
        rle["counts"] = rls["counts"].decode("utf-8")

    coco_results.extend(
        [
            {
                "image_id": original_id,
                "category_id": labels[i],
                "bbox": boxes[i]
                "segmentation": rle,
                "score": scores[i],
            }
            for i, rle in enumerate(rles)
        ]
    )
    return coco_results