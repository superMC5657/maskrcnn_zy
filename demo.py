import os
import torch
import matplotlib.pyplot as plt
import pytorch_mask_rcnn as pmr
from torch.utils.tensorboard import SummaryWriter

use_cuda = True
dataset = "coco"
ckpt_path = "/home1/zhangyan/workplace/maskrcnn_zy/weight/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
data_dir = "/home1/datasets/coco/coco2017/"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
#print("\ndevice:{}".format(device))

ds = pmr.datasets(dataset, data_dir, "val2017", train=False)
indices = torch.randperm(len(ds)).tolist()
d = torch.utils.data.Subset(ds, indices)


model = pmr.maskrcnn_resnet50(True, len(ds.classes) + 1).to(device) # pretrained ds.classes = 80
model_raw = pmr.maskrcnn_resnet50_raw().to(device)
model.eval()
model.head.score_thresh = 0.3


input = torch.rand(3, 640,1312).cuda()
with SummaryWriter(comment='model') as w1:
    w1.add_graph(model, (input,))

