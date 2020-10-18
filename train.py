import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision import transforms


# from .generalized_dataset import GeneralizedDataset

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO:args.use_cuda


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="")

    parser.add_argument("--use-cuda", action="store_true")

    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", default="/data/coco2017")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")

    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[22, 26])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")

    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.02 * 1 / 16  # lr should be 'batch_size / 16 * 0.02' experience formula
