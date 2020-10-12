import torch
import time

from pytorch_mask_rcnn import Meter

try:
    from pytorch_mask_rcnn.datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p['lr'] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()

        losses = model(image, target)
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)

        S = time.time()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        b_m.update(time.time() - S)

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

        A = time.time() - A
        print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters, 1000*t_m.avg, 1000*b_m.avg))
        return A / iters

def evaluate(model, data_loader, device, args, generate=True):
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)



@torch.no_grad()
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
    ann_labels = data_loader.ann_labels

    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)

        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco)