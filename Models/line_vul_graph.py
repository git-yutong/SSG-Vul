# This is Stagedmodel_line_vul_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch


class GGNNLayer(nn.Module):
    """
    更稳的 GGNN 一步（推荐）：
      - PreNorm(h)
      - 消息：m_ij = W * h_j
      - 聚合：对每个节点 i 做 mean 聚合（sum / deg），避免度数造成尺度漂移
      - 更新：GRUCell(agg, h_norm)
      - Residual：h_out = h + Dropout(h_new)
      - PostNorm（可选，这里提供）
    """
    def __init__(self, hidden_dim, dropout=0.1, use_post_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

        self.use_post_norm = use_post_norm
        self.post_norm = nn.LayerNorm(hidden_dim) if use_post_norm else None

    def forward(self, h, edge_index):
        """
        h: [N, H]
        edge_index: [2, E], (row, col) 表示 col -> row 的边
        """
        # PreNorm（更稳）
        h_in = h
        h = self.pre_norm(h)

        row, col = edge_index  # col -> row
        N, Hdim = h.size()

        # 1) 边消息
        m = self.msg(h[col])  # [E, H]

        # 2) 聚合：sum + degree，然后做 mean（避免高度节点放大）
        agg = torch.zeros(N, Hdim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, row, m)  # sum

        deg = torch.zeros(N, device=h.device, dtype=h.dtype)
        deg.index_add_(0, row, torch.ones(row.size(0), device=h.device, dtype=h.dtype))
        agg = agg / (deg.unsqueeze(-1) + 1e-6)  # mean

        # 3) GRU 更新（注意：用归一化后的 h 作为 hidden state）
        h_new = self.gru(agg, h)  # [N, H]

        # 4) Residual + Dropout
        h_out = h_in + F.dropout(h_new, p=self.dropout, training=self.training)

        # 5) PostNorm（可选）
        if self.use_post_norm:
            h_out = self.post_norm(h_out)

        return h_out


class ResidualGCN(nn.Module):
    """
    原来的 ResidualGCN（保留不动）
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.1, use_gru=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru

        if use_gru:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.gru_out_dim = hidden_dim
        else:
            self.gru_out_dim = hidden_dim

        self.fc_func = nn.Sequential(
            nn.Linear(self.gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        self.fc_line = nn.Sequential(
            nn.Linear(self.gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.view(-1)

        h = x
        for conv in self.convs:
            h_res = h
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h_res.size() == h.size():
                h = h + h_res

        if self.use_gru:
            h_dense, mask = to_dense_batch(h, batch)
            lengths = mask.sum(dim=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                h_dense, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            mask = mask[:, :out.size(1)]
            h = out[mask]

        graph_emb = global_mean_pool(h, batch)
        logits_func = self.fc_func(graph_emb)
        logits_line = self.fc_line(h)
        return logits_func, logits_line

class GraphTransGGNN(nn.Module):
    """
    纯图结构版：
      Transformer#1 → GAT → Transformer#2 → GCN → GGNN(T步) →
      行级 head → segment-level attention pooling → 函数级 head
    不再使用 BiGRU。
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3,
                 dropout=0.1, heads=4,
                 transformer_dropout=0.1,
                 num_ggnn_steps=2):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_ggnn_steps = num_ggnn_steps

        # ===== 0) Transformer #1 (全局行语义建模) =====
        self.trans_enc_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=2,
                dropout=transformer_dropout,
                batch_first=True
            ),
            num_layers=1
        )

        # ===== 1) GATConv (带注意力，可使用 edge_attr) =====
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout
        )

        # ===== 2) Transformer #2 (强化全局依赖) =====
        self.trans_enc_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dropout=transformer_dropout,
                batch_first=True
            ),
            num_layers=1
        )

        # ===== 3) GCNConv (结构平滑整合) =====
        self.gcn = GCNConv(hidden_dim, hidden_dim)

        # ===== 3.5) GGNN 多步消息传递 =====
        self.ggnn_layer = GGNNLayer(hidden_dim) if num_ggnn_steps > 0 else None

        # 这里不再有 BiGRU，输出维度就是 hidden_dim
        self.out_dim = hidden_dim

        # ===== 4) 行级 Head =====
        self.fc_line = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # ===== 5) segment-level attention pooling =====
        self.seg_attn = nn.Linear(self.out_dim, 1)

        # ===== 6) 函数级 Head =====
        self.fc_func = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        seg_id = getattr(data, "seg_id", None)

        # edge_attr → [E,1]
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            if edge_weight.dim() == 1:
                edge_attr = edge_weight.view(-1, 1)
            else:
                edge_attr = edge_weight
        else:
            edge_attr = None

        # ===== A. Transformer #1 =====
        x_dense, mask = to_dense_batch(x, batch)      # [B, L, in_dim]
        x_dense = self.trans_enc_1(x_dense)
        h = x_dense[mask]                             # [N_total, in_dim]

        # ===== B. GATConv =====
        h = self.gat(h, edge_index, edge_attr=edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ===== C. Transformer #2 =====
        h_dense, mask = to_dense_batch(h, batch)      # [B, L, hidden_dim]
        h_dense = self.trans_enc_2(h_dense)
        h = h_dense[mask]                             # [N_total, hidden_dim]

        # ===== D. GCN =====
        h = self.gcn(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ===== D.5 GGNN 多步消息传递 =====
        if self.ggnn_layer is not None and self.num_ggnn_steps > 0:
            for _ in range(self.num_ggnn_steps):
                h = self.ggnn_layer(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # ===== E. 行级逐节点分类 =====
        logits_line = self.fc_line(h)          # [N_total, 2]
        line_probs = F.softmax(logits_line, dim=-1)[:, 1].unsqueeze(-1)   # [N_total,1]

        # ===== F. Segment-level pooling 构造函数级表示 =====
        h_dense, mask = to_dense_batch(h, batch)                 # [B, L, H]
        probs_dense, _ = to_dense_batch(line_probs, batch)       # [B, L, 1]

        if seg_id is None:
            # 无 seg 信息时，退回全局 attention pooling
            h_weighted = h * line_probs                          # [N_total,H]
            num = global_add_pool(h_weighted, batch)             # [B, H]
            denom = global_add_pool(line_probs, batch)           # [B, 1]
            graph_emb = num / (denom + 1e-8)                     # [B, H]
        else:
            seg_id_dense, _ = to_dense_batch(seg_id, batch)      # [B, L]
            B, L, H = h_dense.size()
            S = int(seg_id_dense.max().item()) + 1

            seg_emb_list = []
            for s in range(S):
                seg_mask = (seg_id_dense == s) & mask            # [B, L]
                seg_mask_f = seg_mask.float().unsqueeze(-1)      # [B, L, 1]

                hp = h_dense * probs_dense                       # [B, L, H]
                hp = hp * seg_mask_f

                seg_sum = hp.sum(dim=1)                          # [B, H]
                seg_den = (probs_dense * seg_mask_f).sum(dim=1)  # [B, 1]
                seg_den = seg_den + 1e-8

                seg_emb = seg_sum / seg_den                      # [B, H]
                seg_emb_list.append(seg_emb)

            seg_emb_all = torch.stack(seg_emb_list, dim=1)       # [B, S, H]

            attn_scores = self.seg_attn(seg_emb_all).squeeze(-1)         # [B, S]
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, S, 1]
            graph_emb = (seg_emb_all * attn_weights).sum(dim=1)          # [B, H]

        # ===== G. 函数级 head =====
        logits_func = self.fc_func(graph_emb)                     # [B, 2]

        return logits_func, logits_line


class GraphDualBranch(nn.Module):
    """
    双分支结构（结构分支 + 序列分支）再融合

    - 共享输入：先把 in_dim 投影到 hidden_dim
    - 结构分支（structure branch）：
        PreNorm -> GCN * L 层（残差）-> (可选多步 GGNN)
    - 序列分支（sequence branch）：
        Linear -> BiGRU 按 batch/行序建模
    - 融合：
        门控融合：h = z * h_struct + (1 - z) * h_seq
    - 行级 head：
        对融合后的 h 做逐节点分类 logits_line
    - 函数级 head：
        用 logits_line 的正类概率做权重 + segment-level attention pooling
        得到 graph_emb，再做函数级 logits_func
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_gcn_layers=2,
        dropout=0.1,
        use_ggnn=True,
        num_ggnn_steps=2,
        use_seq_branch=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_ggnn = use_ggnn
        self.num_ggnn_steps = num_ggnn_steps
        self.use_seq_branch = use_seq_branch

        # ===== 0) 输入投影到统一维度 =====
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        # ===== Norms（稳定训练，提升可复现与收敛质量）=====
        self.norm_in = nn.LayerNorm(hidden_dim)  # 输入投影后
        self.norm_gcn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)])  # 每层 GCN PreNorm
        self.norm_seq = nn.LayerNorm(hidden_dim)  # 序列分支输出
        self.norm_fuse = nn.LayerNorm(hidden_dim) # 融合后

        # ===== 1) 结构分支：多层 GCN + 可选 GGNN =====
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.ggnn_layer = GGNNLayer(hidden_dim, dropout=dropout, use_post_norm=False) if use_ggnn and num_ggnn_steps > 0 else None

        # ===== 2) 序列分支：BiGRU 按行顺序建模 =====
        if use_seq_branch:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        else:
            self.gru = None

        # ===== 3) 融合门控 =====
        # 输入是 [h_struct, h_seq] 拼接后的 2*H
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),   # 输出 z \in (0,1)^{H}
        )

        # ===== 4) 行级 Head =====
        self.fc_line = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # ===== 5) segment-level attention pooling =====
        self.seg_attn = nn.Linear(hidden_dim, 1)

        # ===== 6) 函数级 Head =====
        self.fc_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        seg_id = getattr(data, "seg_id", None)

        # edge_attr 只在需要时使用（这里 GCN 不需要 edge_weight，只用结构）
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.view(-1)

        # ===== A. 输入投影 =====
        # x: [N, in_dim] -> [N, H]
        h0 = self.in_proj(x)            # [N, H]
        h0 = self.norm_in(h0)           # 稳定尺度

        # ===== B. 结构分支：GCN + 可选 GGNN =====
        h_struct = h0
        for i, conv in enumerate(self.gcn_layers):
            h_res = h_struct

            # PreNorm：先 LN 再进 GCN（更稳）
            h_struct = self.norm_gcn[i](h_struct)

            h_struct = conv(h_struct, edge_index, edge_weight=edge_weight)
            h_struct = F.relu(h_struct)
            h_struct = F.dropout(h_struct, p=self.dropout, training=self.training)

            # Residual
            if h_res.size() == h_struct.size():
                h_struct = h_struct + h_res

        if self.ggnn_layer is not None:
            for _ in range(self.num_ggnn_steps):
                h_struct = self.ggnn_layer(h_struct, edge_index)
                h_struct = F.relu(h_struct)

        # ===== C. 序列分支：BiGRU（按行顺序） =====
        if self.use_seq_branch and self.gru is not None:
            # 序列分支输入用 h0（你也可以改成 h_struct，看你想不想共享图结构信息）
            h_dense, mask = to_dense_batch(h0, batch)     # [B, L, H], mask: [B, L]
            lengths = mask.sum(dim=1)                     # [B]

            packed = nn.utils.rnn.pack_padded_sequence(
                h_dense,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            # out: [B, L', H], L' = 当前 batch 最长长度
            mask = mask[:, :out.size(1)]                  # [B, L']

            # 展平回 [N, H]
            h_seq = out[mask]                             # [N, H]
            h_seq = self.norm_seq(h_seq)                  # 稳定序列分支尺度
        else:
            # 没有序列分支时，退化为只用结构分支（也做一次 norm 以对齐尺度）
            h_seq = self.norm_seq(h_struct)

        # ===== D. 双分支融合：门控加权 + LN =====
        h_cat = torch.cat([h_struct, h_seq], dim=-1)      # [N, 2H]
        z = self.fuse_gate(h_cat)                         # [N, H]
        h = z * h_struct + (1.0 - z) * h_seq              # [N, H]
        h = self.norm_fuse(h)                             # 融合后归一化，利于 line head 稳定

        # ===== E. 行级预测（逐节点） =====
        logits_line = self.fc_line(h)                     # [N, 2]
        line_probs = F.softmax(logits_line, dim=-1)[:, 1].unsqueeze(-1)  # [N, 1]

        # ===== F. segment-level pooling 构造函数级表示 =====
        h_dense, mask = to_dense_batch(h, batch)                 # [B, L, H]
        probs_dense, _ = to_dense_batch(line_probs, batch)       # [B, L, 1]

        if seg_id is None:
            # 无 segment 信息时，退回全局加权平均（权重=行正类概率）
            h_weighted = h * line_probs                          # [N, H]
            num = global_add_pool(h_weighted, batch)             # [B, H]
            denom = global_add_pool(line_probs, batch)           # [B, 1]
            graph_emb = num / (denom + 1e-8)
        else:
            seg_id_dense, _ = to_dense_batch(seg_id, batch)      # [B, L]
            B, L, Hdim = h_dense.size()
            S = int(seg_id_dense.max().item()) + 1

            seg_emb_list = []
            for s in range(S):
                seg_mask = (seg_id_dense == s) & mask            # [B, L]
                seg_mask_f = seg_mask.float().unsqueeze(-1)      # [B, L, 1]

                hp = h_dense * probs_dense                       # [B, L, H]
                hp = hp * seg_mask_f

                seg_sum = hp.sum(dim=1)                          # [B, H]
                seg_den = (probs_dense * seg_mask_f).sum(dim=1)  # [B, 1]
                seg_den = seg_den + 1e-8

                seg_emb = seg_sum / seg_den                      # [B, H]
                seg_emb_list.append(seg_emb)

            seg_emb_all = torch.stack(seg_emb_list, dim=1)       # [B, S, H]

            attn_scores = self.seg_attn(seg_emb_all).squeeze(-1)         # [B, S]
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, S, 1]
            graph_emb = (seg_emb_all * attn_weights).sum(dim=1)          # [B, H]

        # ===== G. 函数级 head =====
        logits_func = self.fc_func(graph_emb)                     # [B, 2]

        return logits_func, logits_line 
