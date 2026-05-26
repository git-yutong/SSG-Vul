# This is util.py
import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm   # ★ 新增：TQDM
import multiprocessing

import torch
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.data import Data
import torch
import os
import multiprocessing
from tqdm import tqdm
import math


import re

KEYWORDS = {
    "if","for","while","switch","case","return",
    "int","char","void","long","short","float","double",
    "static","const","unsigned","signed","struct",
    "class","public","private","protected","try","catch","throw","new",
    "true","false","null","nullptr",
}


def parse_flaw_indices_0based(flaw_str):
    """
    解析 CSV 中的 flaw_line_index，返回 0-based 漏洞行下标列表。
    如果为空或格式异常，返回 []。
    """
    if isinstance(flaw_str, float) and np.isnan(flaw_str):
        return []
    if not isinstance(flaw_str, str):
        return []
    if flaw_str.strip() == "":
        return []

    try:
        return [int(x.strip()) for x in flaw_str.split(",") if x.strip() != ""]
    except Exception:
        return []
    
def is_identifier_like(s: str) -> bool:
    if not s:
        return False
    # 必须含字母或下划线开头（可按语言微调）
    if not re.match(r"^[A-Za-z_]\w*$", s):
        return False
    if s in KEYWORDS:
        return False
    return True

def clean_roberta_token(tok: str) -> str:
    # UniXcoder/Roberta 常见：Ġ 表示词边界
    return tok.replace("Ġ", "")

LOW_INFO_IDENTIFIERS = {
    "i", "j", "k", "n", "m",
    "len", "length", "size",
    "tmp", "temp",
    "data", "buf", "buffer",
    "ret", "res", "result",
    "val", "value",
    "idx", "index",
}


def strip_comments_keep_lines(code: str) -> str:
    """
    移除 // 与 /* */ 注释，但保持：
      - 行数不变（换行保留）
      - 尽量保持列位置（用空格替换注释内容）
    目的：减少注释 token 噪声，同时不破坏 flaw_line_index / valid_mask 的行号对齐。
    """
    if not isinstance(code, str):
        code = str(code)

    out_lines = []
    in_block = False

    for line in code.split("\n"):
        s = line
        i = 0
        res = []
        while i < len(s):
            if in_block:
                end = s.find("*/", i)
                if end == -1:
                    # 整行都在块注释里 -> 全空格
                    res.append(" " * (len(s) - i))
                    i = len(s)
                else:
                    # 注释结束前替换为空格，跳过 */
                    res.append(" " * (end + 2 - i))
                    i = end + 2
                    in_block = False
            else:
                # 查找 // 与 /* 的最早出现
                idx_line = s.find("//", i)
                idx_blk = s.find("/*", i)

                if idx_line == -1 and idx_blk == -1:
                    res.append(s[i:])
                    break

                # 取最近的注释起点
                nxt = min([x for x in [idx_line, idx_blk] if x != -1])

                if nxt > i:
                    res.append(s[i:nxt])
                    i = nxt

                # 处理 // 注释
                if idx_line != -1 and nxt == idx_line:
                    # 从 // 到行末替换为空格
                    res.append(" " * (len(s) - i))
                    i = len(s)
                    break

                # 处理 /* */ 注释
                if idx_blk != -1 and nxt == idx_blk:
                    end = s.find("*/", i + 2)
                    if end == -1:
                        # 从 /* 到行末替换为空格，并进入块注释状态
                        res.append(" " * (len(s) - i))
                        i = len(s)
                        in_block = True
                    else:
                        # 同行闭合：替换这段为空格
                        res.append(" " * (end + 2 - i))
                        i = end + 2

        out_lines.append("".join(res))

    return "\n".join(out_lines)

def is_empty_flaw(s):
    if pd.isna(s):
        return True
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        if isinstance(s, float) and math.isnan(s):
            return True
        return False
    if isinstance(s, str):
        return len(s.strip()) == 0
    return True
def tokenize_func_into_segments(
    func,
    tokenizer,
    block_size=512,
    seg_num=4,
):
    """
    把一个函数按行切开并 tokenize，切成最多 seg_num 段。
    关键约束：**不要在某一行中间切**。

    做法：
      1. 先得到 code_tokens 和 row_idx（每个 token 对应的全局行号 1..num_lines）
      2. 每段最多 block_size-2 个 code token（留 CLS/SEP）
      3. 如果切点落在一行中间，就把这一整行挪到下一段：
         => 当前段的 end_idx = 这一行的起始 token 下标（类似你 StagedVul 里的逻辑）

    返回：
        seg_input_ids_list : List[List[int]]，每段长度 = block_size
        seg_line_ids_list  : List[List[int]]，同样长度，0=CLS/SEP/PAD，其它=全局行号(1..num_lines)
        num_lines          : 原始函数总行数
    """
    rows = str(func).split("\n")
    rows = ["\n" if x == "" else x for x in rows]
    num_lines = len(rows) if len(rows) > 0 else 1

    # 逐行 tokenize
    row_tokens = [tokenizer.tokenize(x) for x in rows]

    code_tokens = []
    row_idx = []  # 和 StagedVul 一样：每个 token 对应的行号（1..num_lines）
    for i, toks in enumerate(row_tokens):
        line_no = i + 1
        for _t in toks:
            code_tokens.append(_t)
            row_idx.append(line_no)

    # 没有任何 token，返回一个空段（基本不会发生）
    if len(code_tokens) == 0:
        pad_id = tokenizer.pad_token_id
        seg_input_ids = [tokenizer.cls_token, tokenizer.sep_token]
        seg_line_ids = [0, 0]
        # padding 到 block_size
        if len(seg_input_ids) < block_size:
            pad_len = block_size - len(seg_input_ids)
            seg_input_ids += [pad_id] * pad_len
            seg_line_ids += [0] * pad_len
        else:
            seg_input_ids = seg_input_ids[:block_size]
            seg_line_ids = seg_line_ids[:block_size]

        return [tokenizer.convert_tokens_to_ids(seg_input_ids)], [seg_line_ids], num_lines

    max_code_per_seg = block_size - 2  # 每段能容纳的 code token 数
    seg_input_ids_list = []
    seg_line_ids_list = []
    pad_id = tokenizer.pad_token_id

    start_idx = 0
    total_tokens = len(code_tokens)

    for _seg in range(seg_num):
        if start_idx >= total_tokens:
            break

        # 先假设这一段拿 max_code_per_seg 个 token
        tentative_end = start_idx + max_code_per_seg
        if tentative_end >= total_tokens:
            # 剩下的 token 全部放进这一段
            end_idx = total_tokens
        else:
            # tentative_end 指向的是“假想的最后一个 token 的下标”（半开区间左闭右开，所以要减 1）
            last_token_idx = tentative_end - 1
            last_row = row_idx[last_token_idx]

            # 在 [start_idx, total_tokens) 范围内，找到 last_row 首次出现的位置，
            # 也就是该行的第一个 token，把它挪到下一段。
            try:
                first_pos_of_last_row = row_idx.index(last_row, start_idx)
            except ValueError:
                # 理论上不会发生，保险起见
                first_pos_of_last_row = last_token_idx

            end_idx = first_pos_of_last_row

            # 极端情况：某一行 token 数本身就 > max_code_per_seg，
            # 会导致 end_idx == start_idx，一直卡死。
            # 这种情况下，只能“破例”在行内截断一次（非常少见）。
            if end_idx == start_idx:
                end_idx = min(start_idx + max_code_per_seg, total_tokens)

        # 防御：如果还是没有前进，就 break 避免死循环
        if end_idx <= start_idx:
            break

        # 构造这一段的 token & line_ids
        seg_tokens = [tokenizer.cls_token] + code_tokens[start_idx:end_idx] + [tokenizer.sep_token]
        seg_line_ids = [0] + row_idx[start_idx:end_idx] + [0]

        input_ids = tokenizer.convert_tokens_to_ids(seg_tokens)

        # padding / 截断到 block_size
        if len(input_ids) < block_size:
            pad_len = block_size - len(input_ids)
            input_ids += [pad_id] * pad_len
            seg_line_ids += [0] * pad_len
        else:
            input_ids = input_ids[:block_size]
            seg_line_ids = seg_line_ids[:block_size]

        seg_input_ids_list.append(input_ids)
        seg_line_ids_list.append(seg_line_ids)

        # 下一段从 end_idx 开始
        start_idx = end_idx

        # 总 token 超过 seg_num 能容纳的范围，就截断掉后面的（按段截）
        if start_idx >= total_tokens:
            break

    return seg_input_ids_list, seg_line_ids_list, num_lines
def build_line_graph_for_func_multi_segment(
    seg_input_ids_list,
    seg_line_ids_list,
    tokenizer,
    word_embeddings,
    window_size=3,
    same_name_weight=3.0,
    weighted_graph=True,
    # ===== Ablation 开关 =====
    use_A1=True,
    use_A2=True,
    use_A3=True,
    # ===== A3 降噪超参 =====
    ident_min_len=2,
    ident_max_occ=8,
    ident_max_line_gap=200,
    # ===== 行特征池化方式 =====
    line_pooling="mean",   # "mean" / "max" / "weight" / "weighted" / "concat"
    pool_info_weight=2.0,
    pool_default_weight=1.0,
    pool_low_weight=0.5,
    # ===== A1 only-valid =====
    valid_mask=None,
):
    """
    行特征池化支持：
      - mean:
          对同一行所有 token embedding 做平均池化。

      - max:
          对同一行所有 token embedding 做 max pooling。

      - weight / weighted:
          对同一行 token embedding 做加权平均。
          信息 token 权重大，普通 token 权重正常，低信息 token / 符号 token 权重较小。

      - concat:
          拼接 mean pooling 和 max pooling。
          line_feat = concat([mean_feat, max_feat])
          因此输出维度是原始 embedding 维度的 2 倍。
    """
    assert len(seg_input_ids_list) == len(seg_line_ids_list)

    if isinstance(word_embeddings, torch.Tensor):
        word_embeddings = word_embeddings.detach().cpu().numpy()

    hidden_dim = word_embeddings.shape[1]

    # ===== 规范化 pooling 参数 =====
    line_pooling = (line_pooling or "mean").lower()
    if line_pooling == "weight":
        line_pooling = "weighted"

    if line_pooling not in ["mean", "max", "weighted", "concat"]:
        raise ValueError(
            f"Unsupported line_pooling={line_pooling}, "
            f"choose from ['mean', 'max', 'weight', 'weighted', 'concat']"
        )

    # ===== 推 num_lines =====
    max_line = 1
    for line_ids in seg_line_ids_list:
        for lid in line_ids:
            if lid > 0:
                max_line = max(max_line, lid)
    num_lines = max_line

    # ===== 对齐 valid_mask 到 num_lines =====
    if valid_mask is not None:
        if len(valid_mask) > num_lines:
            valid_mask = valid_mask[:num_lines]
        elif len(valid_mask) < num_lines:
            valid_mask = valid_mask + [True] * (num_lines - len(valid_mask))
        valid_mask_arr = np.array(valid_mask, dtype=bool)
    else:
        valid_mask_arr = np.ones(num_lines, dtype=bool)

    line_seg_ids = np.full(num_lines, -1, dtype=np.int64)

    # ===== 为不同 pooling 准备累积矩阵 =====
    # mean / concat 都需要 sum
    line_sum_feats = np.zeros((num_lines, hidden_dim), dtype=np.float32)

    # max / concat 都需要 max
    line_max_feats = np.full((num_lines, hidden_dim), -1e9, dtype=np.float32)

    # weighted 需要 weighted sum
    line_weighted_feats = np.zeros((num_lines, hidden_dim), dtype=np.float32)

    line_has_token = np.zeros(num_lines, dtype=bool)
    line_token_count = np.zeros(num_lines, dtype=np.int64)
    line_token_weight_sum = np.zeros(num_lines, dtype=np.float32)

    edge_weight = defaultdict(float)
    pad_ids = {tokenizer.pad_token_id, 1, 2}  # 兼容 RoBERTa: <pad>=1, </s>=2

    # ===== A2 信息量 token 过滤 =====
    def is_informative_token(tok_str: str) -> bool:
        if tok_str is None:
            return False

        t = clean_roberta_token(tok_str).strip()
        if len(t) == 0:
            return False

        # 过滤纯符号
        if not any(ch.isalnum() or ch == "_" for ch in t):
            return False

        tl = t.lower()

        # 过滤关键词、低信息标识符
        if tl in KEYWORDS:
            return False
        if tl in LOW_INFO_IDENTIFIERS:
            return False

        # 过滤太短的 BPE 碎片
        if len(t) < 3:
            return False

        return True

    def get_pool_weight(tok_str: str) -> float:
        """
        weighted pooling 使用的 token 权重：
          - 信息 token：pool_info_weight
          - 普通 token：pool_default_weight
          - 符号 / 低信息 token：pool_low_weight
        """
        if tok_str is None:
            return float(pool_low_weight)

        t = clean_roberta_token(tok_str).strip()
        if len(t) == 0:
            return float(pool_low_weight)

        tl = t.lower()

        # 纯符号，例如 ; { } ( ) = 等
        if not any(ch.isalnum() or ch == "_" for ch in t):
            return float(pool_low_weight)

        # 低信息变量，例如 i, j, len, data, tmp 等
        if tl in LOW_INFO_IDENTIFIERS:
            return float(pool_low_weight)

        # 高信息 token
        if is_informative_token(tok_str):
            return float(pool_info_weight)

        # 关键词、数字、普通 token
        return float(pool_default_weight)

    # ===== 每个 segment 内部构边 =====
    for seg_idx, (seg_input_ids, seg_line_ids) in enumerate(
        zip(seg_input_ids_list, seg_line_ids_list)
    ):
        assert len(seg_input_ids) == len(seg_line_ids)

        # 1) 去掉尾部 PAD/</s>
        end = len(seg_input_ids)
        while end > 0 and seg_input_ids[end - 1] in pad_ids:
            end -= 1

        token_ids = seg_input_ids[:end]
        line_ids = seg_line_ids[:end]

        tokens_str = tokenizer.convert_ids_to_tokens(token_ids)

        # ------------------------------------------------------------
        # I. token_line_idx + 行特征池化
        # ------------------------------------------------------------
        token_line_idx = [-1] * len(line_ids)

        for pos, lid in enumerate(line_ids):
            if lid <= 0:
                continue

            idx0 = lid - 1
            token_line_idx[pos] = idx0

            emb = word_embeddings[int(token_ids[pos])]

            # mean / concat: 累加 sum
            if line_pooling in ["mean", "concat"]:
                line_sum_feats[idx0] += emb

            # max / concat: 更新 max
            if line_pooling in ["max", "concat"]:
                if not line_has_token[idx0]:
                    line_max_feats[idx0] = emb
                else:
                    line_max_feats[idx0] = np.maximum(line_max_feats[idx0], emb)

            # weighted: 加权累加
            if line_pooling == "weighted":
                tok_str = tokens_str[pos] if pos < len(tokens_str) else None
                w_pool = get_pool_weight(tok_str)
                line_weighted_feats[idx0] += emb * w_pool
                line_token_weight_sum[idx0] += w_pool

            line_has_token[idx0] = True
            line_token_count[idx0] += 1

            if line_seg_ids[idx0] < 0:
                line_seg_ids[idx0] = seg_idx

        assert len(token_line_idx) == len(token_ids), \
            f"len(token_line_idx)={len(token_line_idx)} != len(token_ids)={len(token_ids)}"

        # ------------------------------------------------------------
        # II. A2：token 共现窗口 -> 行边
        # ------------------------------------------------------------
        if use_A2:
            T = len(token_ids)
            if T > 0:
                info_mask = [False] * T

                for pos, tok in enumerate(tokens_str):
                    li = token_line_idx[pos]
                    if li < 0:
                        continue

                    # 行无效，不参与 A2
                    if not valid_mask_arr[li]:
                        continue

                    # 只保留 RoBERTa / UniXcoder 中的词首 token，减少 BPE 碎片噪声
                    if (not tok.startswith("Ġ")) and (tok not in ["<s>", "</s>", "<pad>"]):
                        continue

                    if is_informative_token(tok):
                        info_mask[pos] = True

                if T <= window_size:
                    windows = [list(range(T))]
                else:
                    windows = [
                        list(range(i, i + window_size))
                        for i in range(T - window_size + 1)
                    ]

                # 按窗口数归一化，避免长函数边权过大
                num_windows = max(1, len(windows))
                w_norm = 1.0 / float(num_windows)

                for win in windows:
                    info_pos = [p for p in win if info_mask[p]]
                    if len(info_pos) <= 1:
                        continue

                    seen_pairs = set()

                    for ii in range(1, len(info_pos)):
                        for jj in range(0, ii):
                            pi = info_pos[ii]
                            pj = info_pos[jj]

                            li = token_line_idx[pi]
                            lj = token_line_idx[pj]

                            if li < 0 or lj < 0 or li == lj:
                                continue

                            if (not valid_mask_arr[li]) or (not valid_mask_arr[lj]):
                                continue

                            a, b = (li, lj) if li < lj else (lj, li)
                            if (a, b) in seen_pairs:
                                continue

                            seen_pairs.add((a, b))

                            edge_weight[(li, lj)] += w_norm
                            edge_weight[(lj, li)] += w_norm

        # ------------------------------------------------------------
        # III. A3：identifier 同名边
        # ------------------------------------------------------------
        if use_A3:
            id2lines = defaultdict(set)
            cur = ""
            cur_line = -1

            for pos, tok in enumerate(tokens_str):
                li = token_line_idx[pos]

                if li < 0:
                    if is_identifier_like(cur) and cur_line >= 0:
                        id2lines[cur].add(cur_line)
                    cur = ""
                    cur_line = -1
                    continue

                # 行无效，不参与 A3
                if not valid_mask_arr[li]:
                    if is_identifier_like(cur) and cur_line >= 0:
                        id2lines[cur].add(cur_line)
                    cur = ""
                    cur_line = li
                    continue

                t = clean_roberta_token(tok)

                if len(t) == 0 or (not any(ch.isalnum() or ch == "_" for ch in t)):
                    if is_identifier_like(cur) and cur_line >= 0:
                        id2lines[cur].add(cur_line)
                    cur = ""
                    cur_line = li
                    continue

                is_new_word = tok.startswith("Ġ")

                if li != cur_line:
                    if is_identifier_like(cur) and cur_line >= 0:
                        id2lines[cur].add(cur_line)
                    cur = t
                    cur_line = li
                    continue

                if is_new_word:
                    if is_identifier_like(cur) and cur_line >= 0:
                        id2lines[cur].add(cur_line)
                    cur = t
                else:
                    cur = cur + t

            if is_identifier_like(cur) and cur_line >= 0:
                id2lines[cur].add(cur_line)

            for ident, lines_set in id2lines.items():
                ident_lower = ident.lower()

                if ident_lower in LOW_INFO_IDENTIFIERS:
                    continue
                if len(ident) < ident_min_len:
                    continue
                if len(lines_set) <= 1:
                    continue
                if ident_max_occ is not None and len(lines_set) > ident_max_occ:
                    continue

                lines = sorted(lines_set)
                w = float(same_name_weight) / (len(lines) ** 0.5)

                for a in range(len(lines)):
                    for b in range(a + 1, len(lines)):
                        la, lb = lines[a], lines[b]

                        if ident_max_line_gap is not None and abs(la - lb) > ident_max_line_gap:
                            continue

                        edge_weight[(la, lb)] += w
                        edge_weight[(lb, la)] += w

    # ===== 最终生成 line_feats =====
    if line_pooling == "mean":
        safe_count = np.maximum(line_token_count, 1).reshape(-1, 1)
        line_feats = line_sum_feats / safe_count

    elif line_pooling == "max":
        line_feats = line_max_feats

    elif line_pooling == "weighted":
        safe_weight_sum = np.maximum(line_token_weight_sum, 1e-12).reshape(-1, 1)
        line_feats = line_weighted_feats / safe_weight_sum

    elif line_pooling == "concat":
        safe_count = np.maximum(line_token_count, 1).reshape(-1, 1)

        mean_feats = line_sum_feats / safe_count
        max_feats = line_max_feats

        # 无 token 行置零，避免 max_feats 里残留 -1e9
        mean_feats[~line_has_token] = 0.0
        max_feats[~line_has_token] = 0.0

        # concat: [mean, max]
        line_feats = np.concatenate([mean_feats, max_feats], axis=1).astype(np.float32)

    else:
        raise RuntimeError(f"Unexpected line_pooling={line_pooling}")

    # ===== 行特征/seg 防御处理 =====
    line_feats[~line_has_token] = 0.0
    line_seg_ids[line_seg_ids < 0] = 0

    # ------------------------------------------------------------
    # IV. A1：只在“有效行序列”之间连边
    # ------------------------------------------------------------
    if use_A1:
        eff_valid = valid_mask_arr & line_has_token
        valid_lines = np.where(eff_valid)[0].tolist()

        for k in range(len(valid_lines) - 1):
            i = valid_lines[k]
            j = valid_lines[k + 1]

            edge_weight[(i, j)] += 1.0
            edge_weight[(j, i)] += 1.0

    # ===== 邻接矩阵 =====
    rows, cols, vals = [], [], []

    for (u, v), w in edge_weight.items():
        rows.append(u)
        cols.append(v)
        vals.append(float(w if weighted_graph else 1.0))

    if len(rows) == 0:
        adj = sp.csr_matrix((num_lines, num_lines), dtype=np.float32)
    else:
        adj = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(num_lines, num_lines),
            dtype=np.float32,
        )

    return adj, line_feats, num_lines, line_seg_ids


def parse_flaw_line_index(flaw_str, num_lines, valid_mask=None):
    """
    从 CSV 读取 flaw_line_index 作为行标签，注意：
    👉 CSV 中的 flaw_line_index 已经是 0-based 行号（和你之前校验脚本保持一致）。
       即：0 表第 1 行，1 表第 2 行，依此类推。

    我们在这里直接按 0-based 使用：
        labels[idx0] = 1  对应 idx0 ∈ [0, num_lines-1]
    """
    # 1) 解析 0-based 行号
    if isinstance(flaw_str, float) and np.isnan(flaw_str):
        vuln_idx = []
    elif not isinstance(flaw_str, str):
        vuln_idx = []
    elif flaw_str.strip() == "":
        vuln_idx = []
    else:
        try:
            parts = flaw_str.split(",")
            vuln_idx = [int(x.strip()) for x in parts if x.strip() != ""]
        except Exception:
            vuln_idx = []

    # 2) 默认所有行标签先置 0
    labels = [0] * num_lines
    for idx0 in vuln_idx:
        if 0 <= idx0 < num_lines:
            labels[idx0] = 1

    # 3) 应用 valid_mask：无效行 => -1
    if valid_mask is not None:
        assert len(valid_mask) == num_lines
        for i, is_valid in enumerate(valid_mask):
            if not is_valid:
                labels[i] = -1

    return labels


def tokenize_func_with_line_ids(func, tokenizer, max_len=512):
    """
    把一个函数按行切开，然后用 GraphCodeBERT tokenizer 编码，
    同时给每个 token 一个「行号」，行号从 1 开始（0 留给 [CLS]/[SEP]/[PAD]）。

    返回：
        input_ids : 长度 = max_len 的列表
        line_ids  : 长度 = max_len 的列表（0 / 1..num_lines）
        num_lines : 函数的总行数
    """
    rows = str(func).split("\n")
    rows = ["\n" if x == "" else x for x in rows]
    num_lines = len(rows) if len(rows) > 0 else 1

    row_tokens = [tokenizer.tokenize(x) for x in rows]
    # 去掉完全 token 为空的行，但行号仍然从 1..num_lines
    flat_tokens = []
    flat_line_ids = []
    for idx, toks in enumerate(row_tokens):
        line_no = idx + 1  # 1-based
        for t in toks:
            flat_tokens.append(t)
            flat_line_ids.append(line_no)

    max_code_tokens = max_len - 2
    flat_tokens = flat_tokens[:max_code_tokens]
    flat_line_ids = flat_line_ids[:max_code_tokens]

    tokens = [tokenizer.cls_token] + flat_tokens + [tokenizer.sep_token]
    line_ids = [0] + flat_line_ids + [0]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # padding
    pad_id = tokenizer.pad_token_id
    if len(input_ids) < max_len:
        pad_len = max_len - len(input_ids)
        input_ids += [pad_id] * pad_len
        line_ids += [0] * pad_len
    else:
        input_ids = input_ids[:max_len]
        line_ids = line_ids[:max_len]

    return input_ids, line_ids, num_lines


import re

def get_valid_line_mask_from_processed_func(code: str):
    """
    对 processed_func 做行级过滤，返回每一行是否参与节点分类的布尔列表：
      - False（无效）：空行 / 只有 '}' / 注释行 //... / 注释块 /* ... */ 内的行
      - True （有效）：其他代码行（包括带行尾注释的代码，比如: int x; // comment）

    注意：这里按原始 processed_func 的行号来判断，
          这样和 flaw_line_index 的 1-based 行号完全对齐。
    """
    if not isinstance(code, str):
        code = str(code)

    lines = code.split("\n")
    valid = []
    in_block_comment = False

    for line in lines:
        raw = line  # 保留一下原始行（如果以后想更细分用得到）
        stripped = line.strip()

        # 1) 空行 / 只有 '}' 直接无效
        if stripped == "" or stripped == "}":
            valid.append(False)
            # 不改变 block_comment 状态
            continue

        # 2) 已经在注释块里：直到遇到 '*/' 之前的所有行都算注释行
        if in_block_comment:
            valid.append(False)
            if "*/" in stripped:
                in_block_comment = False
            continue

        # 3) 单行注释：// 开头
        if stripped.startswith("//"):
            valid.append(False)
            continue

        # 4) 块注释开始：/* ...  */
        if "/*" in stripped:
            start_pos = stripped.find("/*")
            end_pos = stripped.find("*/", start_pos + 2)

            # (a) 整行从 /* 开始且后面只有注释：视为纯注释行
            if start_pos == 0 and (end_pos == -1 or stripped[end_pos+2:].strip() == ""):
                valid.append(False)
                if end_pos == -1:
                    in_block_comment = True
                # 如果同一行就结束了注释块，不开启 in_block_comment
                continue
            # (b) 代码 + 行尾注释：比如 "int x; /* comment */"
            else:
                # 有代码在 /* 前面 => 这行仍然算代码行
                if end_pos == -1:
                    # 代码 + 开启块注释并跨行
                    in_block_comment = True
                valid.append(True)
                continue

        # 5) 普通代码行
        valid.append(True)

    return valid


from collections import defaultdict
import scipy.sparse as sp

def build_line_graph_for_one(
    token_ids,
    line_ids,
    tokenizer,
    word_embeddings,
    window_size=3,
    same_name_weight=3.0,
    weighted_graph=True,
):
    """
    对单个函数构建行级图：
      - 行向量：用 word_embeddings 查表 + max pooling
      - 边：
          * 共现窗口（A2）
          * 同名标识符连边（A3，带关键词/符号过滤）
          * 多阶行序边（A1：i<->i+1, i<->i+2，按距离衰减）
    """
    pad_ids = {tokenizer.pad_token_id, 1, 2}  # 兼容 RoBERTa 的 <pad>=1, </s>=2

    # 1) 去掉尾部 padding / </s>
    assert len(token_ids) == len(line_ids)
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] in pad_ids:
        end -= 1
    token_ids = token_ids[:end]
    line_ids = line_ids[:end]

    if len(token_ids) == 0:
        hidden_dim = word_embeddings.shape[1]
        adj = sp.csr_matrix((1, 1), dtype=np.float32)
        feats = np.zeros((1, hidden_dim), dtype=np.float32)
        num_lines = 1
        return adj, feats, num_lines

    # 2) 行数
    max_line = max([l for l in line_ids if l > 0] or [1])
    num_lines = max_line
    hidden_dim = word_embeddings.shape[1]

    # 3) 行级 max pooling（embedding 查表）
    line_feats = np.full((num_lines, hidden_dim), -1e9, dtype=np.float32)
    line_has_token = np.zeros(num_lines, dtype=bool)
    token_line_idx = []

    for pos, lid in enumerate(line_ids):
        if lid <= 0:
            token_line_idx.append(-1)
            continue
        idx0 = lid - 1  # 0-based
        token_line_idx.append(idx0)

        emb = word_embeddings[int(token_ids[pos])]  # [H]
        if not line_has_token[idx0]:
            line_feats[idx0] = emb
            line_has_token[idx0] = True
        else:
            line_feats[idx0] = np.maximum(line_feats[idx0], emb)

    # 没有 token 的行设为 0 向量
    line_feats[~line_has_token] = 0.0

    # 4) 行级边：共现窗口 + 同名连接边 + 多阶行序
    edge_weight = defaultdict(float)
    T = len(token_ids)

    # ---------- 4.1 共现窗口（A2：在 token 层滑窗，再映射到行） ----------
    if T <= window_size:
        windows = [list(range(T))]
    else:
        windows = [list(range(i, i + window_size)) for i in range(T - window_size + 1)]

    for win in windows:
        for i in range(1, len(win)):
            for j in range(0, i):
                pi = win[i]
                pj = win[j]
                li = token_line_idx[pi]
                lj = token_line_idx[pj]
                if li < 0 or lj < 0 or li == lj:
                    continue
                edge_weight[(li, lj)] += 1.0
                edge_weight[(lj, li)] += 1.0

    # ---------- 4.2 同名标识符连边（A3：带关键词/符号过滤） ----------
    tokens_str = tokenizer.convert_ids_to_tokens(token_ids)

    # 简单关键词 + 符号过滤（可按需扩展）
    KEYWORDS = {
        "if", "for", "while", "switch", "case", "return",
        "int", "char", "void", "long", "short", "float", "double",
        "static", "const", "unsigned", "signed", "struct",
        ";", "{", "}", "(", ")", ",", "[", "]", "=",
    }

    pos_by_str = defaultdict(list)
    for pos, tok in enumerate(tokens_str):
        li = token_line_idx[pos]
        if li < 0:
            continue

        # 处理 RoBERTa/BPE 前缀：Ġxxx
        clean_tok = tok.lstrip("Ġ")

        # 过滤关键词 / 纯符号
        if clean_tok in KEYWORDS:
            continue
        if len(clean_tok) == 0:
            continue
        if not any(ch.isalnum() or ch == "_" for ch in clean_tok):
            continue

        pos_by_str[clean_tok].append(pos)

    # 只对出现多次的标识符连边
    for tok, poss in pos_by_str.items():
        if len(poss) <= 1:
            continue
        Lp = len(poss)
        for a in range(Lp):
            for b in range(a + 1, Lp):
                pa = poss[a]
                pb = poss[b]
                la = token_line_idx[pa]
                lb = token_line_idx[pb]
                if la < 0 or lb < 0 or la == lb:
                    continue
                edge_weight[(la, lb)] += same_name_weight
                edge_weight[(lb, la)] += same_name_weight

    # ---------- 4.3 多阶行序边（A1：i<->i+1, i<->i+2, 按距离衰减） ----------
    max_hop = 2
    for i in range(num_lines):
        for d in range(1, max_hop + 1):
            j = i + d
            if j >= num_lines:
                break
            w = 1.0 / d  # 距离越远权重越小
            edge_weight[(i, j)] += w
            edge_weight[(j, i)] += w

    # 5) 构造稀疏邻接矩阵
    rows, cols, vals = [], [], []
    for (u, v), w in edge_weight.items():
        rows.append(u)
        cols.append(v)
        vals.append(float(w if weighted_graph else 1.0))

    if len(rows) == 0:
        adj = sp.csr_matrix((num_lines, num_lines), dtype=np.float32)
    else:
        adj = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(num_lines, num_lines),
            dtype=np.float32,
        )

    return adj, line_feats, num_lines





def scipy_to_pyg(adj_csr):
    adj_csr = adj_csr.tocoo()

    indices = np.vstack((adj_csr.row, adj_csr.col))  # shape: [2, E]

    edge_index = torch.from_numpy(indices).long()
    edge_weight = torch.from_numpy(adj_csr.data.astype(np.float32))

    return edge_index, edge_weight


def build_graphs_from_csv(
    csv_path,
    tokenizer,
    emb_matrix,
    max_len=512,
    seg_num=4,
    window_size=3,
    same_name_weight=3.0,
    device=None,
    num_workers=None,
    use_cache=True,
    # ===== Ablation 开关 =====
    use_A1=True,
    use_A2=True,
    use_A3=True,
    weighted_graph=True,
    # ===== 行特征池化方式 =====
    line_pooling="mean",   # "mean" or "max"
):
    """
    基于多段切分的构图：
      - 每个函数按 block_size=max_len 切成至多 seg_num 段
      - 所有段在行级合并成一张图
      - 行标签仍基于原始 processed_func 的全局行号
      - 行特征池化支持 mean / max

    注意：
      - cache 文件名中已经加入 poolmean / poolmax
      - 修改 line_pooling 后会自动生成不同缓存，不会误读旧图
    """
    line_pooling = (line_pooling or "mean").lower()
    if line_pooling not in ["mean", "max","weight"]:
        raise ValueError(
            f"Unsupported line_pooling={line_pooling}, choose from ['mean', 'max','weight']"
        )

    cache_path = csv_path + (
            f".seg{seg_num}_bs{max_len}"
            f".pool{line_pooling}"
            f".A1{int(use_A1)}A2{int(use_A2)}A3{int(use_A3)}"
            f".w{int(weighted_graph)}"
            f".graphs.pt"
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if use_cache and os.path.exists(cache_path):
        print(f"[build_graphs_from_csv] load cached graphs from {cache_path}")
        graphs = torch.load(cache_path, map_location="cpu")
        return graphs

    print(f"[build_graphs_from_csv] build graphs from {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["target"].isin([0, 1])].reset_index(drop=True)

    if "processed_func" not in df.columns:
        raise ValueError(f"{csv_path} 中缺少 processed_func 列，请先准备好。")

    funcs = df["processed_func"].fillna("").tolist()
    targets = df["target"].astype(int).tolist()
    flaw_strs = df["flaw_line_index"].tolist()

    graphs = []

    for i, func in enumerate(
        tqdm(
            funcs,
            total=len(funcs),
            desc=f"Tokenize+build-graph ({os.path.basename(csv_path)})"
        )
    ):
        y_func = targets[i]
        code_str = funcs[i]
        flaw_str = flaw_strs[i]

        # 0) 先算 valid_mask，基于原始 processed_func 的行
        valid_mask_pre = get_valid_line_mask_from_processed_func(code_str)

        # 1) 分段
        # strip_comments_keep_lines 会保持行数不变，因此不会破坏 flaw_line_index 对齐
        clean_func = strip_comments_keep_lines(code_str)

        seg_input_ids_list, seg_line_ids_list, num_lines_src = tokenize_func_into_segments(
            clean_func,
            tokenizer,
            block_size=max_len,
            seg_num=seg_num,
        )

        # 2) 构建行图
        adj, feat, num_lines_eff, line_seg_ids = build_line_graph_for_func_multi_segment(
            seg_input_ids_list,
            seg_line_ids_list,
            tokenizer=tokenizer,
            word_embeddings=emb_matrix,
            window_size=window_size,
            same_name_weight=same_name_weight,
            weighted_graph=weighted_graph,
            use_A1=use_A1,
            use_A2=use_A2,
            use_A3=use_A3,
            line_pooling=line_pooling,
            valid_mask=valid_mask_pre,
        )

        num_lines_graph = feat.shape[0]

        # 3) 对齐 valid_mask 到 num_lines_graph
        valid_mask = valid_mask_pre

        if len(valid_mask) > num_lines_graph:
            valid_mask = valid_mask[:num_lines_graph]
        elif len(valid_mask) < num_lines_graph:
            valid_mask = valid_mask + [True] * (num_lines_graph - len(valid_mask))

        # 4) 行级标签
        # 如果函数级为漏洞样本，但是 flaw_line_index 为空，则行标签全部设为 -1



        # 4) 行级标签
        # 情况 A：target=1 但是 flaw_line_index 为空
        # 这类样本保留，但行级标签全部设为 -1，只参与函数级训练，不参与行级监督
        if y_func == 1 and is_empty_flaw(flaw_str):
            y_line_list = [-1] * num_lines_graph

        else:
            # 解析原始漏洞行号，CSV 中是 0-based
            vuln_idx_raw = parse_flaw_indices_0based(flaw_str)

            # 情况 B：target=1 且有行标签，但是所有漏洞行都因为截断不在当前图中
            # 这类样本直接跳过，不加入 graphs
            if y_func == 1 and len(vuln_idx_raw) > 0:
                kept_vuln_idx = [
                    idx0 for idx0 in vuln_idx_raw
                    if 0 <= idx0 < num_lines_graph
                ]

                if len(kept_vuln_idx) == 0:
                    # 说明该正样本的所有漏洞行都被截断掉了
                    # 保留它会导致 target=1 但图中没有任何正行标签
                    continue

            y_line_list = parse_flaw_line_index(
                flaw_str,
                num_lines_graph,
                valid_mask=valid_mask,
            )


        # 5) 保险：再次对齐 y_line 长度
        if len(y_line_list) > num_lines_graph:
            y_line_list = y_line_list[:num_lines_graph]
        elif len(y_line_list) < num_lines_graph:
            y_line_list = y_line_list + [-1] * (num_lines_graph - len(y_line_list))

        y_line = torch.tensor(y_line_list, dtype=torch.long)

        edge_index, edge_weight = scipy_to_pyg(adj)
        x = torch.tensor(feat, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y_func=torch.tensor(y_func, dtype=torch.long),
            y_line=y_line,
            num_lines=num_lines_graph,
            seg_id=torch.tensor(line_seg_ids, dtype=torch.long),
        )

        # 防止 x / y_line / seg_id 行数不一致
        assert data.x.size(0) == data.y_line.size(0) == data.seg_id.size(0), \
            f"graph {i}: x={data.x.size(0)}, y_line={data.y_line.size(0)}, seg_id={data.seg_id.size(0)}"

        graphs.append(data)

    # 6) 缓存
    if use_cache:
        torch.save(graphs, cache_path)
        print(f"[build_graphs_from_csv] saved graphs to {cache_path}")
    else:
        print("[build_graphs_from_csv] use_cache=False, 不保存缓存图。")

    return graphs
