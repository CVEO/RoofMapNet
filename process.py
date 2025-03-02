#!/usr/bin/env python3
"""Process a dataset with the trained neural network
Usage:
    process.py [options] <yaml-config> <checkpoint> <image-dir> <output-dir>
    process.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <image-dir>                   Path to the directory containing processed images
   <output-dir>                  Path to the output directory

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   --plot                        Plot the result
"""

import os
import sys
import shlex
import pprint
import random
import os.path as osp
import threading
import subprocess

import yaml
import numpy as np
import torch
import matplotlib as mpl
import skimage.io
import matplotlib.pyplot as plt
from docopt import docopt

import roofmapnet
from roofmapnet.utils import recursive_to
from roofmapnet.config import C, M
from roofmapnet.datasets import WireframeDataset, collate
from roofmapnet.models.regression import MultitaskHead
from roofmapnet.models.roofmapnet import RoofMapNet


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    if M.backbone == "stacked_hourglass":
        model = RoofMapNet(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )
    else:
        raise NotImplementedError

    checkpoint = torch.load(args["<checkpoint>"])
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(
        WireframeDataset(args["<image-dir>"], split="valid"),
        shuffle=False,
        batch_size=M.batch_size,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )
    os.makedirs(args["<output-dir>"], exist_ok=True)

    for batch_idx, (image, meta, target) in enumerate(loader):
        with torch.no_grad():
            input_dict = {
                "image": recursive_to(image, device),
                "meta": recursive_to(meta, device),
                "target": recursive_to(target, device),
                "mode": "validation",
            }
            H = model(input_dict)["preds"]
            for i in range(M.batch_size):
                index = batch_idx * M.batch_size + i
                np.savez(
                    osp.join(args["<output-dir>"], f"{index:06}.npz"),
                    **{k: v[i].cpu().numpy() for k, v in H.items()},
                )


if __name__ == "__main__":
    main()
