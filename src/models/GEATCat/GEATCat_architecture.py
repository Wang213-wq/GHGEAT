"""
GEATCat架构
基于 GHGEAT 架构，采用 GNNCat 的输出方式
"""
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from utilities_v2.mol2graph import cat_dummy_graph

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExternalAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, u_in, memory_size):
        super(ExternalAttentionLayer, self).__init__()
        self.memory_size = memory_size
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # 外部记忆矩阵 Mk 和 Mv
        self.Mk = nn.Parameter(torch.randn(memory_size, hidden_dim * 2 + u_in))
        self.Mv = nn.Parameter(torch.randn(memory_size, hidden_dim * 2 + u_in))

    def forward(self, x):
        # x形状: (batch_size, num_nodes, in_features)
        batch_size, num_nodes, _ = x.shape

        # 计算注意力分数，形状变化为 (batch_size, num_nodes, memory_size)
        attn = torch.matmul(x, self.Mk.T)  # 通过Mk将输入映射到外部记忆空间

        # 计算注意力权重，形状变化为 (batch_size, num_nodes, memory_size)
        attn = F.softmax(attn, dim=1)

        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # 归一化

        # 使用注意力权重聚合 Mv，形状变化为 (batch_size, num_nodes, hidden_dim)
        out = torch.matmul(attn, self.Mv)  # 通过Mv从外部记忆空间重构特征

        return out


class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats,
                 edge_hidden_feats=32, num_step_message_passing=1):
        super(MPNNconv, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = gnn.NNConv(
            node_out_feats,
            node_out_feats,
            edge_network,
            aggr='add'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, system_graph):
        node_feats = system_graph.x
        edge_index = system_graph.edge_index
        edge_feats = system_graph.edge_attr

        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            if torch.cuda.is_available():
                node_feats = F.relu(self.gnn_layer(x=node_feats.type(torch.FloatTensor).cuda(),
                                                   edge_index=edge_index.type(torch.LongTensor).cuda(),
                                                   edge_attr=edge_feats.type(torch.FloatTensor).cuda()))
            else:
                node_feats = F.relu(self.gnn_layer(x=node_feats.type(torch.FloatTensor),
                                                   edge_index=edge_index.type(torch.LongTensor),
                                                   edge_attr=edge_feats.type(torch.FloatTensor)))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats


class EdgeModel(torch.nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(v_in * 2 + e_in + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], axis=1)
        return self.edge_mlp(out).to(device)


class NodeModel(torch.nn.Module):
    def __init__(self, v_in, u_in, hidden_dim, memory_size=128, attention_weight=1.0):
        super().__init__()
        self.ext_attention = ExternalAttentionLayer(v_in + u_in, hidden_dim, u_in, memory_size)
        self.attention_weight = attention_weight
        self.input_projection = None
        self._input_proj_dim = None
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_add(edge_attr, edge_index[1], dim=0, dim_size=x.size(0))
        combined_input = torch.cat([x, out, u[batch]], dim=1)
        combined_dim = combined_input.size(1)

        # 添加批次维度，形状变为 (1, num_nodes, v_in + u_in)
        out = combined_input.unsqueeze(0)
        attn_out = self.ext_attention(out)  # 注意力输出
        attn_out = attn_out.squeeze(0)  # 移除批次维度

        # 根据attention_weight调整注意力的使用比例
        if self.attention_weight >= 1.0:
            out = attn_out
        else:
            if self.input_projection is None or self._input_proj_dim != combined_dim:
                self._input_proj_dim = combined_dim
                self.input_projection = nn.Linear(combined_dim, attn_out.size(1)).to(combined_input.device)

            original_proj = self.input_projection(combined_input)

            if self.attention_weight <= 0.0:
                out = original_proj
            else:
                out = self.attention_weight * attn_out + (1.0 - self.attention_weight) * original_proj

        return self.node_mlp(out), None


def ensure_tensor(value):
    """确保值是 tensor，如果是 tuple 则递归解包"""
    while isinstance(value, tuple):
        value = value[0] if len(value) > 0 else None
    if value is None:
        raise ValueError("Cannot extract tensor from tuple")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor, but got {type(value)}")
    return value


class GlobalModel(torch.nn.Module):
    def __init__(self, u_in, hidden_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # 确保所有参数都是 tensor 类型
        while isinstance(x, tuple):
            x = x[0]
        while isinstance(batch, tuple):
            batch = batch[0]
        while isinstance(edge_attr, tuple):
            edge_attr = edge_attr[0]
        while isinstance(u, tuple):
            u = u[0]
        while isinstance(edge_index, tuple):
            edge_index = edge_index[0]

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a tensor, but got {type(x)}")
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"Expected batch to be a tensor, but got {type(batch)}")
        if not isinstance(edge_attr, torch.Tensor):
            raise TypeError(f"Expected edge_attr to be a tensor, but got {type(edge_attr)}")
        if not isinstance(u, torch.Tensor):
            raise TypeError(f"Expected u to be a tensor, but got {type(u)}")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"Expected edge_index to be a tensor, but got {type(edge_index)}")

        # 确保所有参数都在正确的设备上
        x = x.to(device)
        batch = batch.to(device)
        edge_attr = edge_attr.to(device)
        u = u.to(device)
        edge_index = edge_index.to(device)

        node_aggregate = scatter_mean(x, batch, dim=0)
        edge_aggregate = scatter_mean(edge_attr, batch[edge_index[1]], dim=0,
                                      out=edge_attr.new_zeros(node_aggregate.shape))
        out = torch.cat([u, node_aggregate, edge_aggregate], dim=1)
        return self.global_mlp(out).to(device)


class GEATCat(nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim, attention_weight=1.0):
        super(GEATCat, self).__init__()
        self.attention_weight = attention_weight

        self.graphnet1 = gnn.MetaLayer(
            EdgeModel(v_in, e_in, u_in, hidden_dim),
            NodeModel(v_in, u_in, hidden_dim, attention_weight=attention_weight),
            GlobalModel(u_in, hidden_dim)
        )
        self.graphnet2 = gnn.MetaLayer(
            EdgeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
            NodeModel(hidden_dim, hidden_dim, hidden_dim, attention_weight=attention_weight),
            GlobalModel(hidden_dim, hidden_dim)
        )

        self.gnorm1 = gnn.GraphNorm(hidden_dim)
        self.gnorm2 = gnn.GraphNorm(hidden_dim)

        self.pool = global_mean_pool

        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim * 2,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim * 2)

        # MLP unique (采用 GNNCat 的输出方式)
        self.mlp1_unique = nn.Linear(hidden_dim * 4 + 1, hidden_dim * 2)
        self.mlp2_unique = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mlp3_unique = nn.Linear(hidden_dim, 1)

    def generate_sys_graph(self, x, edge_attr, batch_size, n_mols=2):
        src = np.arange(batch_size)
        dst = np.arange(batch_size, n_mols * batch_size)
        self_connection = np.arange(n_mols * batch_size)
        one_way = np.concatenate((src, dst, self_connection))
        other_way = np.concatenate((dst, src, self_connection))
        edge_index = torch.tensor([list(one_way), list(other_way)], dtype=torch.long).to(device)
        sys_graph = Data(x=x.to(device), edge_index=edge_index, edge_attr=edge_attr.to(device))
        return sys_graph

    def forward(self, solvent, solute, T):
        # Molecular descriptors based on MOSCED model
        ap1 = solvent.ap.reshape(-1, 1).to(device)
        ap2 = solute.ap.reshape(-1, 1).to(device)
        bp1 = solvent.bp.reshape(-1, 1).to(device)
        bp2 = solute.bp.reshape(-1, 1).to(device)
        topopsa1 = solvent.topopsa.reshape(-1, 1).to(device)
        topopsa2 = solute.topopsa.reshape(-1, 1).to(device)
        intra_hb1 = solvent.inter_hb.to(device)
        intra_hb2 = solute.inter_hb.to(device)

        u1 = torch.cat((ap1, bp1, topopsa1), axis=1).to(device)
        u2 = torch.cat((ap2, bp2, topopsa2), axis=1).to(device)

        single_node_batch = False
        if solvent.edge_attr.shape[0] == 0 or solute.edge_attr.shape[0] == 0:
            solvent = cat_dummy_graph(solvent)
            solute = cat_dummy_graph(solute)
            u1_dummy = torch.tensor([1, 1, 1]).reshape(1, -1).to(device)
            u2_dummy = torch.tensor([1, 1, 1]).reshape(1, -1).to(device)
            u1 = torch.cat((u1, u1_dummy), axis=0)
            u2 = torch.cat((u2, u2_dummy), axis=0)
            single_node_batch = True

        # Solvent GraphNet
        solvent.x = solvent.x.to(device)
        solvent.edge_index = solvent.edge_index.to(device)
        solvent.edge_attr = solvent.edge_attr.to(device)
        solvent.batch = solvent.batch.to(device)

        solute.x = solute.x.to(device)
        solute.edge_index = solute.edge_index.to(device)
        solute.edge_attr = solute.edge_attr.to(device)
        solute.batch = solute.batch.to(device)

        x1, edge_attr1, u1 = self.graphnet1(solvent.x,
                                            solvent.edge_index,
                                            solvent.edge_attr,
                                            u1,
                                            solvent.batch)
        x1 = ensure_tensor(x1)
        edge_attr1 = ensure_tensor(edge_attr1)
        u1 = ensure_tensor(u1)

        x1 = self.gnorm1(x1, solvent.batch)
        x1, edge_attr1, u1 = self.graphnet2(x1,
                                            solvent.edge_index,
                                            edge_attr1,
                                            u1,
                                            solvent.batch)
        x1 = ensure_tensor(x1)
        edge_attr1 = ensure_tensor(edge_attr1)
        u1 = ensure_tensor(u1)

        x1 = self.gnorm2(x1, solvent.batch)
        xg1 = self.pool(x1, solvent.batch)

        # Solute GraphNet
        x2, edge_attr2, u2 = self.graphnet1(solute.x,
                                            solute.edge_index,
                                            solute.edge_attr,
                                            u2,
                                            solute.batch)
        x2 = ensure_tensor(x2)
        edge_attr2 = ensure_tensor(edge_attr2)
        u2 = ensure_tensor(u2)

        x2 = self.gnorm1(x2, solute.batch)
        x2, edge_attr2, u2 = self.graphnet2(x2,
                                            solute.edge_index,
                                            edge_attr2,
                                            u2,
                                            solute.batch)
        x2 = ensure_tensor(x2)
        edge_attr2 = ensure_tensor(edge_attr2)
        u2 = ensure_tensor(u2)

        x2 = self.gnorm2(x2, solute.batch)
        xg2 = self.pool(x2, solute.batch)

        if single_node_batch:
            xg1 = xg1[:-1, :]
            xg2 = xg2[:-1, :]
            u1 = u1[:-1, :]
            u2 = u2[:-1, :]
            solvent.inter_hb = solvent.inter_hb[:-1]
            solute.inter_hb = solute.inter_hb[:-1]
            batch_size = solvent.y.shape[0] - 1
        else:
            batch_size = solvent.y.shape[0]

        inter_hb = solvent.inter_hb
        node_feat = torch.cat((
            torch.cat((xg1, u1), axis=1),
            torch.cat((xg2, u2), axis=1)), axis=0)
        edge_feat = torch.cat((inter_hb.repeat(2),
                               intra_hb1,
                               intra_hb2)).unsqueeze(1)

        binary_sys_graph = self.generate_sys_graph(x=node_feat,
                                                   edge_attr=edge_feat,
                                                   batch_size=batch_size)

        xg = self.global_conv1(binary_sys_graph)
        xg = torch.cat((xg[0:len(xg) // 2, :], xg[len(xg) // 2:, :]), axis=1)

        # 温度归一化（采用 GNNCat 的方式）
        T = T.x.reshape(-1, 1) + 273.15
        T_min = -60 + 273.15
        T_max = 289.3 + 273.15
        T_norm = (T - T_min) / (T_max - T_min)  # 最大最小归一化

        xg = torch.cat((xg, T_norm), axis=1)

        # 单一MLP输出（采用 GNNCat 的方式）
        output = F.relu(self.mlp1_unique(xg))
        output = F.relu(self.mlp2_unique(output))
        output = self.mlp3_unique(output)

        return output.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

