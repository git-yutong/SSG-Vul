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
    window_size=3,
    same_name_weight=3.0,
    device=None,
    num_workers=None,
    use_cache=True,
):
    """
    单进程构图 + 可选缓存：
      - use_cache=True：优先读取/写入 .graphs.pt 缓存
    emb_matrix: 只用来查表，不跑 encoder。
    """
    cache_path = csv_path + ".graphs.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1) 有缓存 & 开启 use_cache，就直接读
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

    # 2) 函数 -> token_ids / line_ids
    doc_token_ids_list = []
    doc_line_ids_list = []
    num_lines_list = []

    for func in tqdm(funcs, total=len(funcs),
                     desc=f"Tokenizing & line-id mapping ({os.path.basename(csv_path)})"):
        input_ids, line_ids, num_lines = tokenize_func_with_line_ids(
            func, tokenizer, max_len=max_len
        )
        doc_token_ids_list.append(input_ids)
        doc_line_ids_list.append(line_ids)
        num_lines_list.append(num_lines)

    # 3) 用 embedding 矩阵直接查表构建行特征 + 邻接
    x_adj_list, x_feat_list, line_nums = build_line_graph_with_embedding(
        doc_token_ids_list,
        doc_line_ids_list,
        emb_matrix,
        tokenizer=tokenizer,
        window_size=window_size,
        same_name_weight=same_name_weight,
    )

    graphs = []
    for i in tqdm(range(len(funcs)), total=len(funcs),
                  desc=f"Building PyG Data ({os.path.basename(csv_path)})"):
        adj = x_adj_list[i]
        feat = x_feat_list[i]        # [num_lines_eff, hidden_dim]
        num_lines = line_nums[i]

        y_func = targets[i]
        code_str = funcs[i]
        flaw_str = flaw_strs[i]

        # =============== 行过滤 mask（仅代码行保留） ===============
        valid_mask = get_valid_line_mask_from_processed_func(code_str)

        # 和 tokenize 的行数对齐一下（通常会相等）
        if len(valid_mask) != num_lines:
            if len(valid_mask) > num_lines:
                valid_mask = valid_mask[:num_lines]
            else:
                valid_mask = valid_mask + [True] * (num_lines - len(valid_mask))

        # ====== 根据 y_func + flaw_line_index 决定行级标签 ======
        if y_func == 1 and is_empty_flaw(flaw_str):
            # ⭐ 函数级为 vul，但缺失行级标注 -> 行级全部未知（-1）
            y_line_list = [-1] * num_lines
        else:
            # 正常情况：用 flaw_line_index + valid_mask 打 0/1/-1
            y_line_list = parse_flaw_line_index(
                flaw_str,
                num_lines,
                valid_mask=valid_mask
            )

        y_line = torch.tensor(y_line_list, dtype=torch.long)

        edge_index, edge_weight = scipy_to_pyg(adj)
        x = torch.tensor(feat, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y_func=torch.tensor(y_func, dtype=torch.long),
            y_line=y_line,
            num_lines=num_lines,
        )
        graphs.append(data)

    # 4) 缓存
    if use_cache:
        torch.save(graphs, cache_path)
        print(f"[build_graphs_from_csv] saved graphs to {cache_path}")
    else:
        print("[build_graphs_from_csv] use_cache=False, 不保存缓存图。")
    return graphs


def build_line_graph_with_embedding(
    doc_token_ids_list,
    doc_line_ids_list,
    emb_matrix,
    tokenizer,
    window_size=3,
    same_name_weight=3.0,
    weighted_graph=True,
):
    """
    基于【embedding 查表】的行级图构建（批量版）。

    参数：
        doc_token_ids_list : List[List[int]]
            每个函数的 input_ids（含 [CLS]/[SEP]/[PAD]）
        doc_line_ids_list  : List[List[int]]
            每个函数的 line_ids（0 / 1..num_lines）
        emb_matrix         : np.ndarray 或 torch.Tensor
            GraphCodeBERT 的 word_embeddings.weight（已转成 numpy 最好）
        tokenizer          : GraphCodeBERT 的 tokenizer
        window_size        : 共现滑动窗口大小（token 层）
        same_name_weight   : 同名 token 行之间边的权重加成
        weighted_graph     : True 使用频数权重，False 所有边权重=1

    返回：
        x_adj_list   : List[scipy.sparse.csr_matrix]，每个函数一张 (num_lines, num_lines) 邻接
        x_feat_list  : List[np.ndarray]，每个函数一份 (num_lines, hidden_dim) 行特征
        line_nums    : List[int]，每个函数的行数
    """
    assert len(doc_token_ids_list) == len(doc_line_ids_list)

    # 如果传进来的是 torch.Tensor，先转成 numpy，避免反复 .cpu().numpy()
    if isinstance(emb_matrix, torch.Tensor):
        word_embeddings = emb_matrix.detach().cpu().numpy()
    else:
        word_embeddings = emb_matrix

    x_adj_list = []
    x_feat_list = []
    line_nums = []

    for i in tqdm(range(len(doc_token_ids_list)),
                  total=len(doc_token_ids_list),
                  desc="Building line-graphs with embedding"):
        token_ids = doc_token_ids_list[i]
        line_ids = doc_line_ids_list[i]

        adj, feats, num_lines = build_line_graph_for_one(
            token_ids=token_ids,
            line_ids=line_ids,
            tokenizer=tokenizer,
            word_embeddings=word_embeddings,
            window_size=window_size,
            same_name_weight=same_name_weight,
            weighted_graph=weighted_graph,
        )

        x_adj_list.append(adj)
        x_feat_list.append(feats)
        line_nums.append(num_lines)

    return x_adj_list, x_feat_list, line_nums




