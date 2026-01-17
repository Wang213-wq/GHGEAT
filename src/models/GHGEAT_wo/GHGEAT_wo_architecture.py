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
from scr.models.utilities_v2.mol2graph import cat_dummy_graph

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExternalAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim,u_in, memory_size):
        super(ExternalAttentionLayer, self).__init__()
        self.memory_size = memory_size
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # 外部记忆矩阵 Mk 和 Mv
        self.Mk = nn.Parameter(torch.randn(memory_size, hidden_dim * 2 + u_in))
        self.Mv = nn.Parameter(torch.randn(memory_size, hidden_dim * 2 + u_in))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x形状: (batch_size, num_nodes, in_features)
        batch_size,num_nodes, _ = x.shape

        # 计算注意力分数，形状变化为 (batch_size, num_nodes, memory_size)
        attention_scores = torch.matmul(x, self.Mk.T) # 通过Mk将输入映射到外部记忆空间

        # 计算注意力权重，形状变化为 (batch_size, num_nodes, memory_size)
        attention_weights = self.softmax(attention_scores) # 对注意力权重进行归一化

        # 使用注意力权重聚合 Mv，形状变化为 (batch_size, num_nodes, hidden_dim)
        out = torch.matmul(attention_weights, self.Mv) # 通过Mv从外部记忆空间重构特征

        return out


class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats,
                 edge_hidden_feats=32, num_step_message_passing=1):
        super(MPNNconv, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),#(38,38)
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        # edge_network用于处理边特征
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),#（1,32)
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)# (32,1444)
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
        return self.edge_mlp(out).to(device)  # 添加 .to(device)


class NodeModel(torch.nn.Module):
    def __init__(self, v_in, u_in, hidden_dim, memory_size=128, use_external_attention=True, attn_alpha=0.5):
        super().__init__()
        self.use_external_attention = use_external_attention
        self.attn_alpha = attn_alpha
        self.ext_attention = ExternalAttentionLayer(v_in + hidden_dim + u_in, hidden_dim, u_in, memory_size) if use_external_attention else None
        self.pre_norm = nn.LayerNorm(v_in + hidden_dim + u_in)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_add(edge_attr, edge_index[1], dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        out = self.pre_norm(out)
        if self.use_external_attention and self.ext_attention is not None:
            attn_input = out.unsqueeze(0)
            attn_out = self.ext_attention(attn_input).squeeze(0)
            out = self.attn_alpha * attn_out + (1.0 - self.attn_alpha) * out
        return self.node_mlp(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, u_in, hidden_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        node_aggregate = scatter_mean(x, batch, dim=0)
        edge_aggregate = scatter_mean(edge_attr, batch[edge_index[1]], dim=0,
                                      out=edge_attr.new_zeros(node_aggregate.shape))
        out = torch.cat([u, node_aggregate, edge_aggregate], dim=1)
        return self.global_mlp(out).to(device)  # 添加 .to(device)


class GHGEAT_wo(nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim, use_external_attention=True, attn_alpha=0.5, memory_size=128):
        super(GHGEAT_wo, self).__init__()

        self.graphnet1 = gnn.MetaLayer(
            EdgeModel(v_in, e_in, u_in, hidden_dim),
            NodeModel(v_in, u_in, hidden_dim, memory_size=memory_size, use_external_attention=use_external_attention, attn_alpha=attn_alpha),
            GlobalModel(u_in, hidden_dim)
        )
        self.graphnet2 = gnn.MetaLayer(
            EdgeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
            NodeModel(hidden_dim, hidden_dim, hidden_dim, memory_size=memory_size, use_external_attention=use_external_attention, attn_alpha=attn_alpha),
            GlobalModel(hidden_dim, hidden_dim)
        )

        self.gnorm1 = gnn.GraphNorm(hidden_dim)
        self.gnorm2 = gnn.GraphNorm(hidden_dim)

        self.pool = global_mean_pool

        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim * 2,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim * 2)

        # MLP for A
        self.mlp1a = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mlp2a = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3a = nn.Linear(hidden_dim, 1)

        # MLP for B
        self.mlp1b = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mlp2b = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3b = nn.Linear(hidden_dim, 1)
    # 生成混合物的结构图
    def generate_sys_graph(self, x, edge_attr, batch_size, n_mols=2): # n_mols指的是分子的个数,指的是溶质和溶剂的分子
        src = np.arange(batch_size) #边的源节点
        dst = np.arange(batch_size, n_mols * batch_size) #边的目标节点
        self_connection = np.arange(n_mols * batch_size) #自环
        one_way = np.concatenate((src, dst, self_connection)) # 将源节点,目标节点,自环合并成为单向边
        other_way = np.concatenate((dst, src, self_connection)) # 反向边
        edge_index = torch.tensor([list(one_way), list(other_way)], dtype=torch.long).to(device) # 图中所有边的连接信息
        sys_graph = Data(x=x.to(device), edge_index=edge_index, edge_attr=edge_attr.to(device))
        return sys_graph

    def forward(self, solvent, solute, T):
        # 分别获取溶质和溶剂的原子极化率ap
        ap1 = solvent.ap.reshape(-1, 1).to(device)
        ap2 = solute.ap.reshape(-1, 1).to(device)
        # 分别获取溶质和溶剂的键极化率bp
        bp1 = solvent.bp.reshape(-1, 1).to(device)
        bp2 = solute.bp.reshape(-1, 1).to(device)
        # 分别获取溶质和溶剂的拓扑极性表面积
        topopsa1 = solvent.topopsa.reshape(-1, 1).to(device)
        topopsa2 = solute.topopsa.reshape(-1, 1).to(device)
        # 分别获取溶质和溶剂的分子间氢键相互作用
        intra_hb1 = solvent.inter_hb.to(device)
        intra_hb2 = solute.inter_hb.to(device)
        # 全局特征:原子极化率+键极化率+拓扑极性表面积
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
        x1 = self.gnorm1(x1, solvent.batch)
        x1, edge_attr1, u1 = self.graphnet2(x1,
                                            solvent.edge_index,
                                            edge_attr1,
                                            u1,
                                            solvent.batch)
        x1 = self.gnorm2(x1, solvent.batch)
        xg1 = self.pool(x1, solvent.batch)

        # Solute GraphNet
        x2, edge_attr2, u2 = self.graphnet1(solute.x,
                                            solute.edge_index,
                                            solute.edge_attr,
                                            u2,
                                            solute.batch)
        x2 = self.gnorm1(x2, solute.batch)
        x2, edge_attr2, u2 = self.graphnet2(x2,
                                            solute.edge_index,
                                            edge_attr2,
                                            u2,
                                            solute.batch)
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

        xg = self.global_conv1(binary_sys_graph) # 使用基于MPNNConv的图神经网络模块
        xg = torch.cat((xg[0:len(xg) // 2, :], xg[len(xg) // 2:, :]), axis=1)

        T = T.x.reshape(-1, 1) + 273.15

        A = F.relu(self.mlp1a(xg))
        A = F.relu(self.mlp2a(A))
        A = self.mlp3a(A)

        B = F.relu(self.mlp1b(xg))
        B = F.relu(self.mlp2b(B))
        B = self.mlp3b(B)

        output = A + B / T

        return output.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)