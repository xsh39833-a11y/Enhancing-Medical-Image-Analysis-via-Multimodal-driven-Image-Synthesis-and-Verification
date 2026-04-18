import argparse
import os
import sys

import numpy as np
import tqdm
from PIL import Image
from tabulate import tabulate

from uacanet_utils.eval_functions import EnhancedMeasure, Fmeasure_calu, StructureMeasure, original_WFb
from uacanet_utils.utils import load_config, parse_args


def evaluate(opt, args):
    os.makedirs(opt.Eval.result_path, exist_ok=True)
    method = os.path.split(opt.Eval.pred_root)[-1]
    thresholds = np.linspace(1, 0, 256)
    headers = opt.Eval.metrics
    results = []
    datasets = tqdm.tqdm(opt.Eval.datasets, desc="Evaluation") if args.verbose else opt.Eval.datasets

    for dataset in datasets:
        pred_root = os.path.join(opt.Eval.pred_root, dataset)
        gt_root = os.path.join(opt.Eval.gt_root, dataset, "masks")
        preds = sorted(os.listdir(pred_root))
        gts = sorted(os.listdir(gt_root))

        threshold_e = np.zeros((len(preds), len(thresholds)))
        threshold_f = np.zeros((len(preds), len(thresholds)))
        threshold_iou = np.zeros((len(preds), len(thresholds)))
        threshold_sen = np.zeros((len(preds), len(thresholds)))
        threshold_spe = np.zeros((len(preds), len(thresholds)))
        threshold_dic = np.zeros((len(preds), len(thresholds)))
        s_measure = np.zeros(len(preds))
        w_fmeasure = np.zeros(len(preds))
        mae_values = np.zeros(len(preds))

        iterator = tqdm.tqdm(enumerate(zip(preds, gts)), total=len(preds), leave=False) if args.verbose else enumerate(zip(preds, gts))
        for i, (pred, gt) in iterator:
            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))
            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            gt_mask = gt_mask.astype(np.float64)
            if gt_mask.max() > 1.0:
                gt_mask = gt_mask / 255.0
            gt_mask = (gt_mask > 0.5).astype(np.float64)
            pred_mask = pred_mask.astype(np.float64) / 255.0

            s_measure[i] = StructureMeasure(pred_mask, gt_mask)
            w_fmeasure[i] = original_WFb(pred_mask, gt_mask)
            mae_values[i] = np.mean(np.abs(gt_mask - pred_mask))

            for j, threshold in enumerate(thresholds):
                _, rec, spe, dic, f_val, iou = Fmeasure_calu(pred_mask, gt_mask, threshold)
                bi_pred = np.zeros_like(pred_mask)
                bi_pred[pred_mask >= threshold] = 1
                threshold_e[i, j] = EnhancedMeasure(bi_pred, gt_mask)
                threshold_f[i, j] = f_val
                threshold_sen[i, j] = rec
                threshold_spe[i, j] = spe
                threshold_dic[i, j] = dic
                threshold_iou[i, j] = iou

        meanEm = np.mean(np.mean(threshold_e, axis=0))
        maxEm = np.max(np.mean(threshold_e, axis=0))
        meanSen = np.mean(np.mean(threshold_sen, axis=0))
        maxSen = np.max(np.mean(threshold_sen, axis=0))
        meanSpe = np.mean(np.mean(threshold_spe, axis=0))
        maxSpe = np.max(np.mean(threshold_spe, axis=0))
        meanDic = np.mean(np.mean(threshold_dic, axis=0))
        maxDic = np.max(np.mean(threshold_dic, axis=0))
        meanIoU = np.mean(np.mean(threshold_iou, axis=0))
        maxIoU = np.max(np.mean(threshold_iou, axis=0))
        Sm = np.mean(s_measure)
        wFm = np.mean(w_fmeasure)
        mae = np.mean(mae_values)

        out = []
        for metric in opt.Eval.metrics:
            out.append(eval(metric))
        results.append([dataset, *out])

        csv_path = os.path.join(opt.Eval.result_path, f"result_{dataset}.csv")
        with open(csv_path, "w", encoding="utf-8") as csv:
            csv.write(", ".join(["method", *headers]) + "\n")
            csv.write(method + "," + ",".join(f"{metric:.4f}" for metric in out) + "\n")

    tab = tabulate(results, headers=["dataset", *headers], floatfmt=".3f")
    if args.verbose:
        print(tab)
    return tab


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    evaluate(opt, args)
