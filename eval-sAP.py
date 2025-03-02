#!/usr/bin/env python3
"""Evaluate sAP5, sAP10, sAP15 for LCNN
Usage:
    eval-sAP.py <path>...
    eval-sAP.py (-h | --help )

Examples:
    python eval-sAP.py logs/*/npz/000*

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
"""

import os
import sys
import glob
import os.path as osp

import numpy as np
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
from docopt import docopt

import roofmapnet.utils
import roofmapnet.metric

GT = r"data/valid/*.npz"


def line_score(path, threshold=5):
    preds = sorted(glob.glob(path))
    gts = sorted(glob.glob(GT))

    n_gt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    for pred_name, gt_name in zip(preds, gts):
        with np.load(pred_name) as fpred:
            lcnn_line = fpred["lines"][:, :, :2]
            lcnn_score = fpred["score"]
        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]
        n_gt += len(gt_line)

        for i in range(len(lcnn_line)):
            if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
                lcnn_line = lcnn_line[:i]
                lcnn_score = lcnn_score[:i]
                break

        tp, fp = roofmapnet.metric.msTPFP(lcnn_line, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)
    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt
    return roofmapnet.metric.ap(lcnn_tp, lcnn_fp)

if __name__ == "__main__":

    def work(path):
        print(f"Working on {path}")
        return [100 * line_score(f"{path}/*.npz", t) for t in [5, 10, 15]]

    def list_subdirectories(folder_path):
        subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        return subdirectories
    dirs = r"results"
    subdirectories = list_subdirectories(dirs)
    for directory in subdirectories:
        results = work(os.path.join(dirs,directory))
        print(results)
