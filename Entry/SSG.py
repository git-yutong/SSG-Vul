# this is line_vul_graph.py
from __future__ import absolute_import, division, print_function

# ===== 标准库 =====
import os
import sys
import math
import random
import pickle
import logging
import multiprocessing
from datetime import datetime
from collections import Counter

# ===== 第三方库 =====
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import argparse

from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModel,
)

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


from Entry.util import build_graphs_from_csv
from Models.line_vul_graph import GraphDualBranch

# ===== logger =====
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-spline-conv'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.metrics import precision_recall_curve

@torch.no_grad()
def search_best_line_threshold_on_val(
    args,
    model,
    loader,
    metric="iou",
    t_min=0.0,
    t_max=1.0,
    num_steps=101,
):
    """
    在 val 上搜索行级阈值 t_line。
    默认：最大化 IoU（只在 y_func==1 的函数上统计，口径与 evaluate/evaluate_on_test 一致）。
    返回：best_t_line, meta
    """
    device = args.device
    model.eval()

    thresholds = np.linspace(t_min, t_max, num_steps, dtype=np.float32)

    # 累积每个阈值下的 IoU 总和与计数
    iou_sum = np.zeros_like(thresholds, dtype=np.float64)
    iou_cnt = np.zeros_like(thresholds, dtype=np.int64)

    for batch in loader:
        batch = batch.to(device)
        logits_func, logits_line = model(batch)

        batch_index = batch.batch.detach().cpu().numpy()  # [N_nodes]
        y_func_batch = batch.y_func.detach().cpu().numpy().astype(int)  # [B]

        line_prob = torch.softmax(logits_line, dim=-1)[:, 1].detach().cpu().numpy()  # [N_nodes]
        line_labels_np = batch.y_line.view(-1).detach().cpu().numpy().astype(int)   # [N_nodes]

        num_graphs = y_func_batch.shape[0]
        for g in range(num_graphs):
            if int(y_func_batch[g]) != 1:
                continue

            mask = (batch_index == g)
            if not np.any(mask):
                continue

            score_g = line_prob[mask]     # [n_g]
            true_g  = line_labels_np[mask]# [n_g]

            # 向量化：对每个阈值计算 pred
            # pred_mat: [T, n_g]
            pred_mat = (score_g[None, :] > thresholds[:, None]).astype(np.int32)

            true_mat = (true_g[None, :] == 1).astype(np.int32)

            inter = (pred_mat & true_mat).sum(axis=1)  # [T]
            union = ((pred_mat == 1) | (true_mat == 1)).sum(axis=1)  # [T]

            # union==0 -> IoU=1
            iou = np.where(union > 0, inter / (union + 1e-12), 1.0)

            iou_sum += iou
            iou_cnt += 1

    avg_iou = np.where(iou_cnt > 0, iou_sum / iou_cnt, 0.0)

    best_idx = int(np.argmax(avg_iou))
    best_t = float(thresholds[best_idx])
    best_score = float(avg_iou[best_idx])

    meta = {
        "metric": metric,
        "best_iou": best_score,
        "num_funcs_used": int(iou_cnt[0]) if len(iou_cnt) > 0 else 0,
        "search_grid": int(num_steps),
        "t_range": [float(t_min), float(t_max)],
    }
    logger.info(f"[VAL line threshold search] best_t_line={best_t:.4f}, best_iou={best_score:.4f}, meta={meta}")
    return best_t, meta


@torch.no_grad()
def search_best_func_threshold_on_val(args, model, loader, metric="f1"):
    """
    在 val 上搜索函数级阈值 t_func。
    默认：最大化 F1（更适合类别不平衡）。
    返回：best_t, meta
    """
    device = args.device
    model.eval()

    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits_func, _ = model(batch)
        prob_pos = torch.softmax(logits_func, dim=-1)[:, 1].detach().cpu().numpy()  # [B]
        labels = batch.y_func.detach().cpu().numpy().astype(int)                   # [B]
        all_probs.append(prob_pos)
        all_labels.append(labels)

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # PR curve 上搜索 best F1
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # 注意：precision/recall 比 thresholds 多一个点，对齐时取[:-1]
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)

    if len(thresholds) == 0:
        # 极端情况：全同类等
        best_t = 0.5
        best_f1 = 0.0
    else:
        best_idx = int(np.argmax(f1))
        best_t = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx])

    meta = {
        "metric": metric,
        "best_f1": float(best_f1),
        "num_samples": int(len(labels)),
        "pos_ratio": float(labels.mean()) if len(labels) > 0 else 0.0,
    }
    logger.info(f"[VAL func threshold search] best_t_func={best_t:.4f}, best_f1={best_f1:.4f}, meta={meta}")
    return best_t, meta


@torch.no_grad()
def evaluate_on_test(args, model, loader, split_name="test", func_threshold=0.5, line_threshold=0.1):
    """
    专用于 test：
    - 函数级：用 func_threshold 对正类概率做阈值判定（而不是 argmax）
    - 行级：用 line_threshold 做 IoU / 行级指标阈值（保持你原逻辑）
    """
    device = args.device
    model.eval()

    all_func_probs = []    # 收集函数级正类概率 p(y=1)
    all_func_labels = []   # 收集函数级标签 y_func

    IoUs = 0.0
    IoUs_num = 0

    top5_percent_hits = 0
    top10_percent_hits = 0

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0

    num_vul_funcs_for_acc = 0
    num_vul_tp_for_percent = 0

    for batch in loader:
        batch = batch.to(device)
        logits_func, logits_line = model(batch)

        # ===== 函数级：prob + threshold =====
        func_prob_batch = torch.softmax(logits_func, dim=-1)[:, 1].detach().cpu().numpy()  # [B]
        func_label_batch = batch.y_func.detach().cpu().numpy().astype(int)                  # [B]
        func_pred_batch = (func_prob_batch >= func_threshold).astype(int)                  # [B]

        all_func_probs.append(func_prob_batch)
        all_func_labels.append(func_label_batch)

        # ===== 行级（保持你原来的口径）=====
        batch_index = batch.batch.detach().cpu().numpy()                    # [N_nodes]
        y_func_batch = func_label_batch                                     # [B]

        line_prob = torch.softmax(logits_line, dim=-1)[:, 1].detach().cpu().numpy()  # [N_nodes]
        line_labels_np = batch.y_line.view(-1).detach().cpu().numpy().astype(int)   # [N_nodes]

        num_graphs = y_func_batch.shape[0]
        for g in range(num_graphs):
            # 只在函数级真实标签 == 1 的样本上统计行级指标
            if int(y_func_batch[g]) != 1:
                continue

            mask = (batch_index == g)
            if not np.any(mask):
                continue

            score_g = line_prob[mask]
            true_g = line_labels_np[mask]

            # IoU：阈值来自 line_threshold
            pred_g = (score_g > line_threshold).astype(int)

            union = np.logical_or(pred_g == 1, true_g == 1).sum()
            inter = np.logical_and(pred_g == 1, true_g == 1).sum()

            if union > 0:
                IoUs += inter / union
            else:
                IoUs += 1.0
            IoUs_num += 1

            num_vul_funcs_for_acc += 1

            order0 = np.argsort(score_g)[::-1]
            n_lines = len(order0)

            # percent top-k
            k5_percent = max(1, math.ceil(0.05 * n_lines))
            top5p_idx = order0[:k5_percent]
            k10_percent = max(1, math.ceil(0.10 * n_lines))
            top10p_idx = order0[:k10_percent]

        
            num_vul_tp_for_percent += 1
            if np.any(true_g[top5p_idx] == 1):
                top5_percent_hits += 1
            if np.any(true_g[top10p_idx] == 1):
                    top10_percent_hits += 1

            # abs top-k
            k1 = min(1, n_lines)
            k3 = min(3, n_lines)
            k5 = min(5, n_lines)
            k10 = min(10, n_lines)

            if np.any(true_g[order0[:k1]] == 1):  top1_hits += 1
            if np.any(true_g[order0[:k3]] == 1):  top3_hits += 1
            if np.any(true_g[order0[:k5]] == 1):  top5_hits += 1
            if np.any(true_g[order0[:k10]] == 1): top10_hits += 1

    # ===== 函数级整体指标（用阈值）=====
    func_probs = np.concatenate(all_func_probs, axis=0)
    func_labels = np.concatenate(all_func_labels, axis=0).astype(int)
    func_preds = (func_probs >= func_threshold).astype(int)

    eval_acc = accuracy_score(func_labels, func_preds)
    eval_rec = recall_score(func_labels, func_preds, zero_division=0)
    eval_pre = precision_score(func_labels, func_preds, zero_division=0)
    eval_f1 = f1_score(func_labels, func_preds, zero_division=0)

    # ===== 行级整体指标 =====
    eval_IoU = IoUs / IoUs_num if IoUs_num > 0 else 0.0

    if num_vul_tp_for_percent > 0:
        eval_top_5_percent_acc = top5_percent_hits / num_vul_tp_for_percent
        eval_top_10_percent_acc = top10_percent_hits / num_vul_tp_for_percent
    else:
        eval_top_5_percent_acc = 0.0
        eval_top_10_percent_acc = 0.0

    if num_vul_funcs_for_acc > 0:
        eval_top_1_acc = top1_hits / num_vul_funcs_for_acc
        eval_top_3_acc = top3_hits / num_vul_funcs_for_acc
        eval_top_5_acc = top5_hits / num_vul_funcs_for_acc
        eval_top_10_acc = top10_hits / num_vul_funcs_for_acc
    else:
        eval_top_1_acc = eval_top_3_acc = eval_top_5_acc = eval_top_10_acc = 0.0

    result = {
        "eval_accuracy": float(eval_acc),
        "eval_recall": float(eval_rec),
        "eval_precision": float(eval_pre),
        "eval_f1": float(eval_f1),
        "eval_IoU": float(eval_IoU),
        "top_5%_accuracy": float(eval_top_5_percent_acc),
        "top_10%_accuracy": float(eval_top_10_percent_acc),
        "top_1_accuracy": float(eval_top_1_acc),
        "top_3_accuracy": float(eval_top_3_acc),
        "top_5_accuracy": float(eval_top_5_acc),
        "top_10_accuracy": float(eval_top_10_acc),
    }

    logger.info(f"[{split_name}] func_threshold={func_threshold:.4f}, line_threshold={line_threshold:.4f}")
    logger.info("    Accuracy          = %.4f", result["eval_accuracy"])
    logger.info("    Precision         = %.4f", result["eval_precision"])
    logger.info("    Recall            = %.4f", result["eval_recall"])
    logger.info("    F1                = %.4f", result["eval_f1"])
    logger.info("   ***************************")
    logger.info("    IoU               = %.4f", result["eval_IoU"])
    logger.info("    Top_5%%_accuracy   = %.4f", result["top_5%_accuracy"])
    logger.info("    Top_10%%_accuracy  = %.4f", result["top_10%_accuracy"])
    logger.info("    Top_1_accuracy    = %.4f", result["top_1_accuracy"])
    logger.info("    Top_3_accuracy    = %.4f", result["top_3_accuracy"])
    logger.info("    Top_5_accuracy    = %.4f", result["top_5_accuracy"])
    logger.info("    Top_10_accuracy   = %.4f", result["top_10_accuracy"])

    return result


def set_seed(args):
    seed = int(args.seed)

    # —— Python / Numpy / PyTorch 层面 ——
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # —— cuDNN 确定性设置 ——
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # —— cuBLAS 是必须的，否则触发 GCNConv/Linear 报错 ——
    # 必须在 import torch 前设也可以，这里也可以设置（在 torch.use_deterministic_algorithms 前）
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # —— 强制 PyTorch 报告非确定性算子（可 debug） ——
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass



def seed_worker(worker_id):
    # PyTorch 会把 initial_seed() 传进来（不同 worker 不同）
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def focal_loss_with_beta(
    logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
    reduction="mean",
):
    """
    logits: [N,2]
    targets: [N]
    beta: focusing 参数（越大越关注 hard positive）
    """
    ce = F.cross_entropy(logits, targets, reduction="none")  # [N]
    pt = torch.exp(-ce)

    # focal: (1-pt)^beta * CE
    loss = ((1 - pt) ** beta) * ce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def compute_loss(logits_func, logits_line, data, alpha=1.0,beta=0.95):
    y_func = data.y_func
    y_line = data.y_line.view(-1)

    # ===== 函数级：Focal Loss（由 β 控制）=====
    loss_func = focal_loss_with_beta(
        logits_func,
        y_func,
        beta=float(beta),
    )

    # ===== 行级（保持 CrossEntropy，不动）=====
    valid_mask = (y_line >= 0)
    if valid_mask.sum() > 0:
        loss_line = F.cross_entropy(
            logits_line[valid_mask],
            y_line[valid_mask]
        )
    else:
        loss_line = torch.tensor(0.0, device=logits_line.device)

    # α：仍然控制行级 loss 权重
    loss = loss_func + alpha * loss_line
    return loss, loss_func.item(), loss_line.item()




from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

@torch.no_grad()
def evaluate(args, model, loader, split_name="eval", best_threshold=0.5):
    """
    函数级（所有样本）:
        - eval_accuracy
        - eval_precision
        - eval_recall
        - eval_f1

    行级（仅 vulnerable 函数 y_func=1 且有正样本的函数）:
        - eval_IoU
        - Top-5%_accuracy   (按百分比：top-5% 行内是否命中至少 1 行漏洞)
        - Top-10%_accuracy
        - Top-1_accuracy    (按行数：top-1 行内是否命中至少 1 行漏洞)
        - Top-3_accuracy
        - Top-5_accuracy
        - Top-10_accuracy
    """
    device = args.device
    model.eval()

    # ------------------------------------------------
    # 1. 函数级：累积所有函数的 logits_func 和 y_func
    # ------------------------------------------------
    all_func_logits = []   # [num_funcs_total, 2]
    all_func_labels = []   # [num_funcs_total]

    # ------------------------------------------------
    # 2. 行级（仅 y_func == 1 且有行级正样本的函数）
    # ------------------------------------------------
    IoUs = 0.0
    IoUs_num = 0

    # 百分比版 top-k 命中（和你原来的 top_5% / top_10% 一样）
    top5_percent_hits = 0
    top10_percent_hits = 0

    # 绝对行数版 Top-1/3/5/10 命中
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0

    # 只在 “有行级正样本” 的 vul 函数上统计 top-k accuracy
    num_vul_funcs_for_acc = 0
    num_vul_tp_for_percent = 0

    for batch in loader:
        batch = batch.to(device)

        logits_func, logits_line = model(batch)
        # logits_func: [B, 2]
        # logits_line: [N_batch_nodes, 2]

        # ------ 1) 函数级累积 ------
        all_func_logits.append(logits_func.cpu())
        all_func_labels.append(batch.y_func.cpu())

        # 函数级预测
        func_pred_batch = torch.argmax(logits_func, dim=-1).cpu().numpy()  # [B]

        # ------ 2) 行级处理，仅对 y_func==1 的函数计算 ------
        batch_index = batch.batch.cpu().numpy()                # [N_batch_nodes] 图编号 0..B-1
        y_func_batch = batch.y_func.cpu().numpy()              # [B]

        # 行级概率（正类概率）
        line_prob = torch.softmax(logits_line, dim=-1)[:, 1]   # [N_batch_nodes]
        line_prob_np = line_prob.cpu().numpy()

        line_labels_np = batch.y_line.view(-1).cpu().numpy().astype(int)  # [N_batch_nodes], 0/1

        num_graphs = y_func_batch.shape[0]

        for g in range(num_graphs):
            # 只在函数级标签 == 1 的样本上统计行级指标
            if int(y_func_batch[g]) != 1:
                continue

            mask = (batch_index == g)
            if not np.any(mask):
                continue

            prob_g = line_prob_np[mask]    # [n_g]，正类概率
            true_g = line_labels_np[mask]  # [n_g], 0/1

            # ========================
            # 2.1 IoU（对该函数）
            #     —— 把“没有行索引/没有正样本”的样本也算进去，
            #        并视为“行上没有漏洞”
            # ========================
            score_g = prob_g
            pred_g = (score_g > best_threshold).astype(int)  # 0/1 预测

            union = np.logical_or(pred_g == 1, true_g == 1).sum()
            inter = np.logical_and(pred_g == 1, true_g == 1).sum()

            if union > 0:
                # 有预测正或真实正
                IoUs += inter / union
            else:
                # union == 0：真实和预测都全 0（完全负样本）
                # 视作 IoU=1（完美预测“没有行漏洞”）
                IoUs += 1.0
            IoUs_num += 1

            # ===== 注意：不再跳过没有正样本行的函数 =====
            # 只要 y_func=1，统统计入 Top-k 的分母
            num_vul_funcs_for_acc += 1

            # ========================
            # 2.2 Top-k（按得分排序）
            # ========================
            order0 = np.argsort(score_g)[::-1]  # 0-based 从大到小
            n_lines = len(order0)

            # ---------- 百分比版：Top-5% / Top-10% ----------
            k5_percent = max(1, math.ceil(0.05 * n_lines))
            top5p_idx = order0[:k5_percent]

            k10_percent = max(1, math.ceil(0.10 * n_lines))
            top10p_idx = order0[:k10_percent]

            # 所有 y_func == 1 的函数都参与 Top-k% 计算（无论函数级预测是否正确）
            num_vul_tp_for_percent += 1  
            if np.any(true_g[top5p_idx] == 1):
                top5_percent_hits += 1
            if np.any(true_g[top10p_idx] == 1):
                top10_percent_hits += 1



            # ---------- 行数版：Top-1 / 3 / 5 / 10 ----------
            k1 = min(1, n_lines)
            k3 = min(3, n_lines)
            k5 = min(5, n_lines)
            k10 = min(10, n_lines)

            top1_idx = order0[:k1]
            top3_idx = order0[:k3]
            top5_idx = order0[:k5]
            top10_idx = order0[:k10]

            if np.any(true_g[top1_idx] == 1):
                top1_hits += 1
            if np.any(true_g[top3_idx] == 1):
                top3_hits += 1
            if np.any(true_g[top5_idx] == 1):
                top5_hits += 1
            if np.any(true_g[top10_idx] == 1):
                top10_hits += 1

    # ------------------------------------------------
    # 3. 函数级整体指标（所有函数 vul + non-vul）
    # ------------------------------------------------
    func_logits = torch.cat(all_func_logits, dim=0).numpy()              # [num_funcs_total, 2]
    func_labels = torch.cat(all_func_labels, dim=0).numpy().astype(int)  # [num_funcs_total]

    func_preds = np.argmax(func_logits, axis=-1)  # 0/1

    eval_acc = accuracy_score(func_labels, func_preds)
    eval_rec = recall_score(func_labels, func_preds, zero_division=0)
    eval_pre = precision_score(func_labels, func_preds, zero_division=0)
    eval_f1 = f1_score(func_labels, func_preds, zero_division=0)

    # ------------------------------------------------
    # 4. 行级整体指标
    # ------------------------------------------------
    eval_IoU = IoUs / IoUs_num if IoUs_num > 0 else 0.0

    if num_vul_tp_for_percent > 0:
        eval_top_5_percent_acc = top5_percent_hits / num_vul_tp_for_percent
        eval_top_10_percent_acc = top10_percent_hits / num_vul_tp_for_percent
    else:
        eval_top_5_percent_acc = 0.0
        eval_top_10_percent_acc = 0.0

    # 行数版 Top-1/3/5/10：保持原本逻辑，分母仍然是所有 vul 且有正行的函数
    if num_vul_funcs_for_acc > 0:
        eval_top_1_acc = top1_hits / num_vul_funcs_for_acc
        eval_top_3_acc = top3_hits / num_vul_funcs_for_acc
        eval_top_5_acc = top5_hits / num_vul_funcs_for_acc
        eval_top_10_acc = top10_hits / num_vul_funcs_for_acc
    else:
        eval_top_1_acc = 0.0
        eval_top_3_acc = 0.0
        eval_top_5_acc = 0.0
        eval_top_10_acc = 0.0

    # ------------------------------------------------
    # 5. 汇总结果
    # ------------------------------------------------
    result = {
        # 函数级（vul + non-vul）
        "eval_accuracy": float(eval_acc),
        "eval_recall": float(eval_rec),
        "eval_precision": float(eval_pre),
        "eval_f1": float(eval_f1),

        # 行级（仅 vul 且有正样本的函数）
        "eval_IoU": float(eval_IoU),

        # 百分比版 top-k
        "top_5%_accuracy": float(eval_top_5_percent_acc),
        "top_10%_accuracy": float(eval_top_10_percent_acc),

        # 行数版 top-k
        "top_1_accuracy": float(eval_top_1_acc),
        "top_3_accuracy": float(eval_top_3_acc),
        "top_5_accuracy": float(eval_top_5_acc),
        "top_10_accuracy": float(eval_top_10_acc),

    }
 
    # logging
    logger.info("    Accuracy          = %.4f", result["eval_accuracy"])
    logger.info("    Precision         = %.4f", result["eval_precision"])
    logger.info("    Recall            = %.4f", result["eval_recall"])
    logger.info("    F1                = %.4f", result["eval_f1"])
    logger.info("   ***************************")
    logger.info("    IoU               = %.4f", result["eval_IoU"])
    logger.info("    Top_5%%_accuracy   = %.4f", result["top_5%_accuracy"])
    logger.info("    Top_10%%_accuracy  = %.4f", result["top_10%_accuracy"])
    logger.info("    Top_1_accuracy    = %.4f", result["top_1_accuracy"])
    logger.info("    Top_3_accuracy    = %.4f", result["top_3_accuracy"])
    logger.info("    Top_5_accuracy    = %.4f", result["top_5_accuracy"])
    logger.info("    Top_10_accuracy   = %.4f", result["top_10_accuracy"])
    return result
def calculate_top_k_recall(labels_top_k, sort_ids, top_k):
    top_k_recall = 0
    for idx, x in enumerate(labels_top_k):
        for row in x:
            k = max(1, math.ceil(top_k * len(sort_ids[idx])))
            if row in sort_ids[idx][:k]:
                top_k_recall += 1
                break
    return top_k_recall / len(labels_top_k)

def main():

    parser = argparse.ArgumentParser()
    # parameters（全部保留）
    parser.add_argument("--train_data_file", default="./resource/dataset/source_dataset/train.csv", type=str,           
                        required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--eval_data_file", default="./resource/dataset/source_dataset/val.csv", type=str,
                        help="Validation data file (a csv file).")
    parser.add_argument("--test_data_file", default="./resource/dataset/source_dataset/test.csv", type=str,
                        help="Test data file (a csv file).")
    parser.add_argument("--output_dir", default="./graph_model/", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--pretrained_model",type=str,default="microsoft/graphcodebert-base")
    # parser.add_argument("--pretrained_model",type=str,default="microsoft/unixcoder-base")
    parser.add_argument("--pretrained_model",type=str,default="microsoft/unixcoder-base-nine")
    
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--train_batch_size", type=int, default=192)
    parser.add_argument("--eval_batch_size", type=int, default=192)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of update steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--evaluate_during_training",  default=True, 
                        help="Run evaluation during training at each epoch.")

    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--seg_num", type=int, default=4,
                        help="最大段数，每段约 block_size 的 token，总有效长度 ≈ block_size * seg_num")
    
    parser.add_argument("--do_train", default=True, 
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=True, 
                        help="Whether to run eval on the test set.")

    parser.add_argument('--seed', type=int, default=12345,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=70,
                        help="training epochs")
    
    parser.add_argument("--early_stop_number", type=int, default=40,
                        help="If >0, stop training when eval_f1 does not improve for N consecutive epochs.")
    parser.add_argument("--num_workers", type=int, default=16,
                    help="DataLoader num_workers.")

    parser.add_argument("--line_loss_weight", type=float, default=0.3, 
                        help="Weight of line-level loss when combining with func-level loss.")
    parser.add_argument("--use_A1", type=int, default=0,
                        help="Use A1 sequential line edges (1=yes, 0=no).")
    parser.add_argument("--use_A2", type=int, default=0,
                        help="Use A2 token co-occurrence edges (1=yes, 0=no).")
    parser.add_argument("--use_A3", type=int, default=0,
                        help="Use A3 identifier same-name edges (1=yes, 0=no).")
    parser.add_argument("--gpu_id",type=int,default=0,
                        help="GPU id to use. -1 for CPU, 0 for cuda:0, 1 for cuda:1, etc.")
    parser.add_argument("--line_pooling",type=str,default="mean",choices=["mean", "max","weight"],
    help="Pooling method for aggregating token embeddings into line embeddings."
)
    args = parser.parse_args()


    # Setup CUDA, GPU
    if args.gpu_id == -1:
        device = torch.device("cpu")

    elif args.gpu_id == 0:
        device = torch.device("cuda:0")

    elif args.gpu_id == 1:
        device = torch.device("cuda:1")

    elif args.gpu_id == 2:
        device = torch.device("cuda:2")

    elif args.gpu_id == 3:
        device = torch.device("cuda:3")

    else:
        device = torch.device("cpu")

    args.device = device
    args.n_gpu = 1 
    
    # ====== 创建 run timestamp 子目录 ======
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # ====== log 仍然放在 output_dir 根目录，不进子目录 ======
    log_name = f"log_{timestamp}.txt"
    logging.basicConfig(
        filename=os.path.join(args.output_dir, log_name),
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    logger.info(f"device: {device}")
    logger.info("Graph-SG-VUL Line-level Training")
    logger.info(f"Args: {args}")
    set_seed(args)
    exp_tag = f"A1{args.use_A1}_A2{args.use_A2}_A3{args.use_A3}"
    logger.info(f"Graph ablation setting: {exp_tag}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    base_model = AutoModel.from_pretrained(args.pretrained_model)
    word_embeddings = base_model.get_input_embeddings().weight.data.cpu().numpy()
    hidden_dim = word_embeddings.shape[1]
    del base_model
    torch.cuda.empty_cache()

    build_num_workers = None
    train_graphs = build_graphs_from_csv(
        args.train_data_file,
        tokenizer,
        word_embeddings,
        max_len=args.block_size,      # 每段的长度（含CLS/SEP）
        seg_num=args.seg_num,         # ★ 新增
        window_size=3,
        same_name_weight=3.0,
        num_workers=build_num_workers,
        use_cache=True,
        use_A1=bool(args.use_A1),
        use_A2=bool(args.use_A2),
        use_A3=bool(args.use_A3),
        line_pooling=args.line_pooling
    ) if args.do_train else []

    eval_graphs = build_graphs_from_csv(
        args.eval_data_file,
        tokenizer,
        word_embeddings,
        max_len=args.block_size,
        seg_num=args.seg_num,         # ★ 新增
        window_size=3,
        same_name_weight=3.0,
        num_workers=build_num_workers,
        use_cache=True,
        use_A1=bool(args.use_A1),
        use_A2=bool(args.use_A2),
        use_A3=bool(args.use_A3),
        line_pooling=args.line_pooling
    ) if args.do_train and args.evaluate_during_training else []

    test_graphs = build_graphs_from_csv(
        args.test_data_file,
        tokenizer,
        word_embeddings,
        max_len=args.block_size,
        seg_num=args.seg_num,         # ★ 新增
        window_size=3,
        same_name_weight=3.0,
        num_workers=build_num_workers,
        use_cache=True,
        use_A1=bool(args.use_A1),
        use_A2=bool(args.use_A2),
        use_A3=bool(args.use_A3),
        line_pooling=args.line_pooling
    ) if args.do_test else []

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = PyGDataLoader(
        train_graphs,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,   # 可以 > 0 仍可复现
        pin_memory=True,
        worker_init_fn=seed_worker,     # ★ 必需
        generator=g,                    # ★ 控制 shuffle
    )

    eval_loader = PyGDataLoader(
        eval_graphs,
        # test_graphs,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    ) if eval_graphs else None

    test_loader = PyGDataLoader(
        test_graphs,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    ) if test_graphs else None

    # ===== 计算训练集正负比例，自动生成 focal beta =====
    num_pos = sum([int(g.y_func.item() == 1) for g in train_graphs])
    num_neg = len(train_graphs) - num_pos
    pos_ratio = num_pos / max(1, len(train_graphs))      # ~5%
    # focal β：正样本越少，focal 越强
    raw_beta = 0.25 / max(pos_ratio, 1e-6)   # 正样本 5% → 0.25/0.05=5
    beta = min(5.0, max(1.0, raw_beta))
    logger.info(f"Focal beta (auto): {beta:.4f}   (pos_ratio={pos_ratio:.4f})")
    args.focal_beta = beta

    # === 3) 模型输入维度，用图里的 x 自动推 ===
    in_dim = train_graphs[0].x.size(-1) if train_graphs else hidden_dim
    model = GraphDualBranch(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_gcn_layers=args.num_layers,   # 复用你的 num_layers 做 GCN 层数
        dropout=args.dropout,
        use_ggnn=True,
        num_ggnn_steps=2,                 # 先试 2 步 GGNN，后面可以调参
        use_seq_branch=True,              # 打开序列分支
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon
    )
    scheduler = None

    if args.do_train and len(train_loader) > 0:
        # 每个 epoch 的 “更新步数”（考虑梯度累积）
        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        args.max_steps = args.epochs * steps_per_epoch

        # 如果没有手动给 warmup_steps，就默认用 10%
        warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.1 * args.max_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.max_steps
        )
        logger.info(f"Total training steps: {args.max_steps}, warmup_steps: {warmup_steps}")

    best_f1 = 0.0
    best_ckpt = os.path.join(run_dir, "checkpoint-best-func_f1.pt")

    # >>> 新增：用于保存 Top_10%_accuracy 最高且 F1 >= F1_GATE 的模型 <<<
    F1_GATE = 0.90
    best_top10_acc_under_f1_constraint = -1.0
    best_ckpt_top10 = os.path.join(run_dir, "checkpoint-best-top10_acc_under_f1.pt")

    # ====== 早停（按 eval_f1，不被 Top10 保存逻辑干扰）======
    no_improve_f1_epochs = 0

    global_best_t = 0.5
    fixed_threshold = 0.1
    if args.do_train and train_loader is not None:
        global_step = 0
        # 用 tqdm 包一层 epoch 进度条
        epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", position=0)
        for epoch in epoch_bar:
            model.train()
            epoch_loss = 0.0
            tr_nb = 0

            logger.info(f"******** Epoch {epoch}/{args.epochs} ********")

            # 这里再用一个 tqdm 显示 batch 级进度
            step_bar = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Train Epoch {epoch}",
                position=1,
                leave=False
            )

            for step, batch in enumerate(step_bar):
                batch = batch.to(device)
                logits_func, logits_line = model(batch)
                loss, loss_func, loss_line = compute_loss(
                    logits_func, logits_line, batch,
                    alpha=args.line_loss_weight,
                    beta=args.focal_beta,
                )

                if args.n_gpu > 1:
                    loss = loss.mean()

                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item()
                tr_nb += 1

                # 在 tqdm 进度条上显示当前平均 loss
                avg_loss = epoch_loss / max(1, tr_nb)
                step_bar.set_postfix(loss=f"{avg_loss:.4f}")

            avg_loss = epoch_loss / max(1, tr_nb)
            logger.info(f"Epoch {epoch} finished. train_loss={avg_loss:.5f}")

            if args.evaluate_during_training and eval_loader is not None:
                # 使用固定行级阈值 0.1（只影响 IoU，不影响 top-k% 排序指标）
                eval_result = evaluate(
                    args, model, eval_loader,
                    split_name="eval", best_threshold=fixed_threshold
                )

                func_f1 = eval_result["eval_f1"]

                # ====== 按 eval_f1 保存 best_ckpt ======
                if func_f1 > best_f1:
                    best_f1 = func_f1
                    no_improve_f1_epochs = 0
                    torch.save(model.state_dict(), best_ckpt)
                    logger.info(f"New best eval_f1={best_f1:.4f}, saved to {best_ckpt}")
                else:
                    no_improve_f1_epochs += 1

                # ====== 保存 Top_10%_accuracy 最高且 F1 >= F1_GATE 的模型 ======
                # 若 top10% 更高 -> 保存
                # 若 top10% 相等，但 F1 更高 -> 也保存 仍然要求 F1 >= F1_GATE 才进入该逻辑
                current_top10_acc = eval_result["top_10%_accuracy"]
                if func_f1 >= F1_GATE and (
                    (current_top10_acc > best_top10_acc_under_f1_constraint) or
                    (current_top10_acc == best_top10_acc_under_f1_constraint and func_f1 > best_f1_under_top10_constraint)
                ):
                    best_top10_acc_under_f1_constraint = current_top10_acc
                    best_f1_under_top10_constraint = func_f1
                    torch.save(model.state_dict(), best_ckpt_top10)
                    logger.info(
                        f"New best top_10%_accuracy (or tie+better F1) under F1>={F1_GATE:.2f}: "
                        f"top10_acc={current_top10_acc:.4f}, F1={func_f1:.4f}, "
                        f"saved to {best_ckpt_top10}"
                    )

                # ====== 早停判断（按 eval_f1，与日志一致）======
                if args.early_stop_number > 0 and no_improve_f1_epochs >= args.early_stop_number:
                    logger.info(
                        f"Early stopping triggered: no improvement in eval_f1 for "
                        f"{args.early_stop_number} consecutive epochs."
                    )
                    break

        # 如果训练期间没有做过验证（evaluate_during_training=False），那就直接把最后一版模型保存一下
        if not args.evaluate_during_training:
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"Training finished without eval, saving final model to {best_ckpt}")

    if args.do_test and test_loader is not None:
        logger.info("======= Testing =======")

        # 1) 先加载 best checkpoint（F1 最优）
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            logger.info(f"Loaded best checkpoint from {best_ckpt}")
        else:
            logger.warning(f"No best checkpoint found at {best_ckpt}, using current model weights.")

        # 2) 用该 checkpoint 在 val 上搜索“函数级阈值”和“行级阈值”
        if eval_loader is not None:
            best_t_func, meta_f = search_best_func_threshold_on_val(args, model, eval_loader, metric="f1")
            logger.info(f"Use best_t_func={best_t_func:.4f} for test evaluation. meta={meta_f}")

            best_t_line, meta_l = search_best_line_threshold_on_val(
                args, model, eval_loader,
                metric="iou",
                t_min=0.0, t_max=1.0, num_steps=101
            )
            logger.info(f"Use best_t_line={best_t_line:.4f} for test evaluation. meta={meta_l}")
        else:
            logger.warning("eval_loader is None, fall back to defaults for thresholds.")
            best_t_func = 0.5
            best_t_line = fixed_threshold  # 你原来的 0.1

        # 3) 用阈值在 test 上 evaluate（F1 最优 ckpt）
        _ = evaluate_on_test(
            args, model, test_loader,
            split_name="test",
            func_threshold=best_t_func,
            line_threshold=best_t_line
        )

        # 4) 评估一下 “Top10-under-F1” ckpt（不影响你原逻辑）
        if os.path.exists(best_ckpt_top10):
            logger.info("======= Testing (Top10-under-F1 ckpt) =======")
            model.load_state_dict(torch.load(best_ckpt_top10, map_location=device))
            _ = evaluate_on_test(
                args, model, test_loader,
                split_name="test_top10",
                func_threshold=best_t_func,
                line_threshold=best_t_line
            )
        else:
            logger.warning(f"No best top10 checkpoint found at {best_ckpt_top10}, skip test_top10.")

if __name__ == "__main__":
    main()
