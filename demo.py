# -*- coding: utf-8 -*-
# !@time: 2020/10/18 下午7:54
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py

import os
import torch
import matplotlib.pyplot as plt
import pytorch_mask_rcnn as pmr

use_cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dataset = "coco"

data_dir = "/home1/datasets/coco/coco2017"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice:{}".format(device))

ds = pmr.datasets(dataset, data_dir, "val2017", train=False)
indices = torch.randperm(len(ds)).tolist()
d = torch.utils.data.Subset(ds, indices)

model = pmr.maskrcnn_resnet50(True, len(ds.classes) + 11).to(device)
model.eval()
model.head.score_thresh = 0.3


iters = 3

for i, (image, target) in enumerate(d):
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)
    if result['labels'] is not None:
        plt.figure(figsize=(12, 15))
        pmr.show(image, result, ds.classes)

        if i >= iters - 1:
            break
