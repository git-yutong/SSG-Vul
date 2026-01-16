# This is Stagedmodel_line_vul_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch


class ResidualGCN(nn.Module):
    """
    带残差的多层 GCN + BiGRU：
      - GCN 先做图结构建模（行节点图）
      - BiGRU 再按 “行序” 做一遍序列建模（同一函数的节点顺序=行号顺序）
      - 输出：
          logits_func: [B, 2]     函数级二分类
          logits_line: [N_total, 2]  行级分类（逐节点=逐行）
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

        # ===== BiGRU，输入/输出维度都等于 hidden_dim =====
        if use_gru:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,   # 双向 → 输出维度 hidden_dim
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.gru_out_dim = hidden_dim
        else:
            self.gru_out_dim = hidden_dim

        # 函数级 head
        self.fc_func = nn.Sequential(
            nn.Linear(self.gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # 行级 head
        self.fc_line = nn.Sequential(
            nn.Linear(self.gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        # x: [N_total, in_dim]
        # edge_index: [2, E]
        # edge_attr: [E] 或 [E, 1]，作为 GCNConv 的 edge_weight
        # batch: [N_total]，每个节点对应的图编号 0..B-1
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None and edge_weight.dim() > 1:
            # 如果是 [E,1] 之类，压成 [E]
            edge_weight = edge_weight.view(-1)

        # ===== 1. 多层 GCN + 残差 =====
        h = x
        for conv in self.convs:
            h_res = h
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h_res.size() == h.size():
                h = h + h_res  # 残差

        # ===== 2. BiGRU：按“行顺序”对每个函数做序列建模 =====
        if self.use_gru:
            # to_dense_batch 会把每个图的节点打包成 [B, max_nodes, H]
            # 其中 padding 的部分 mask=False
            h_dense, mask = to_dense_batch(h, batch)   # h_dense: [B, max_nodes, H]
            lengths = mask.sum(dim=1)                  # [B] 每个图真实节点数

            # pack_padded_sequence 需要长度在 CPU 上
            packed = nn.utils.rnn.pack_padded_sequence(
                h_dense, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            # 通过 BiGRU
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            # out: [B, L_max', H_gru]，L_max' = 当前 batch 内最长的真实长度

            # mask 也裁到同样的长度
            mask = mask[:, :out.size(1)]          # [B, L_max']
            # 把 [B, L_max', H] 再展平回 [N_total, H]，顺序跟原始节点一致
            h = out[mask]                         # [N_total, H_gru]

        # ===== 3. 函数级：对每个图做 mean pooling =====
        graph_emb = global_mean_pool(h, batch)   # [B, H_gru]
        logits_func = self.fc_func(graph_emb)    # [B, 2]

        # ===== 4. 行级：逐节点分类（节点 = 行） =====
        logits_line = self.fc_line(h)            # [N_total, 2]

        return logits_func, logits_line





class GraphTransGRU(nn.Module):

    def __init__(self, in_dim, hidden_dim=128, num_layers=3,
                 dropout=0.1, use_gru=True, heads=4, transformer_dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru
        self.heads = heads

        # ===== 0) Transformer #1 (全局行语义建模) =====
        self.trans_enc_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=1,
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
            edge_dim=1,           # edge_attr=[E,1]
            dropout=dropout
        )

        # ===== 2) Transformer #2 (强化全局依赖) =====
        self.trans_enc_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=1,
                dropout=transformer_dropout,
                batch_first=True
            ),
            num_layers=1
        )

        # ===== 3) GCNConv (结构平滑整合) =====
        self.gcn = GCNConv(hidden_dim, hidden_dim)

        # ===== 4) BiGRU 行序列建模 =====
        if use_gru:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
            self.gru_out_dim = hidden_dim
        else:
            self.gru_out_dim = hidden_dim

        # ===== 5) 函数级 Head =====
        self.fc_func = nn.Sequential(
            nn.Linear(self.gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # ===== 6) 行级 Head =====
        self.fc_line = nn.Sequential(
            nn.Linear(self.gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

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
        x_dense = self.trans_enc_1(x_dense)           # Transformer
        x = x_dense[mask]                             # 回到 [N_total, in_dim]

        # ===== B. GATConv =====
        h = self.gat(x, edge_index, edge_attr=edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ===== C. Transformer #2 =====
        h_dense, mask = to_dense_batch(h, batch)      # [B, L, hidden_dim]
        h_dense = self.trans_enc_2(h_dense)
        h = h_dense[mask]

        # ===== D. GCN =====
        h = self.gcn(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ===== E. BiGRU =====
        if self.use_gru:
            h_dense, mask = to_dense_batch(h, batch)
            lengths = mask.sum(dim=1)

            packed = nn.utils.rnn.pack_padded_sequence(
                h_dense,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            mask = mask[:, :out.size(1)]
            h = out[mask]          # [N_total, H]

        # ===== F. 行级逐节点分类（先算行，用作注意力权重） =====
        logits_line = self.fc_line(h)          # [N_total, 2]

        # 取正类（vulnerable）概率作为注意力权重
        line_probs = F.softmax(logits_line, dim=-1)[:, 1]   # [N_total]
        line_probs = line_probs.unsqueeze(-1)               # [N_total, 1]

        # ===== G. Attention Readout 做函数级聚合 =====
        # 先做加权特征 h * w
        h_weighted = h * line_probs                         # [N_total, H]

        # 对每个图做 sum(w * h) 和 sum(w)
        num = global_add_pool(h_weighted, batch)            # [B, H]
        denom = global_add_pool(line_probs, batch)          # [B, 1]

        # 防止除 0，加一个很小的 eps
        graph_emb = num / (denom + 1e-8)                    # [B, H]

        # 函数级 head
        logits_func = self.fc_func(graph_emb)               # [B, 2]

        return logits_func, logits_line

