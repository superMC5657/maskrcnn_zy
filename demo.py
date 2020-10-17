import os
import torch
import matplotlib.pyplot as plt
import pytorch_mask_rcnn as pmr

use_cuda = True
dataset = "coco"
ckpt_path = "/home1/zhangyan/workplace/maskrcnn_zy/weight/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
data_dir = "/home1/datasets/coco/coco2017/"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice:{}".format(device))

ds = pmr.datasets(dataset, data_dir, "val2017", train=False)
indices = torch.randperm(len(ds)).tolist()
d = torch.utils.data.Subset(ds, indices)

model = pmr.maskrcnn_resnet50(True, len(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

print(os.path.expandvars(ckpt_path))
if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(checkpoint["eval_info"])
    model.load_state_dict(checkpoint)

    del checkpoint
    torch.cuda.empty_cache()

for p in model.parameters():
    p.requires_grad_(False)

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