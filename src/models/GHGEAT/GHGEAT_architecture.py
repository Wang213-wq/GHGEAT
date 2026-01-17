import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_add, scatter_mean

from scr.models.utilities_v2.mol2graph import cat_dummy_graph

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
        # 注意：不在此处实例化 nn.Softmax；在 forward 中使用 F.softmax(attn, dim=1)

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
            nn.Linear(node_in_feats, node_out_feats),  # (38,38)
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),  # (1,32)
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)  # (32,1444)
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
    def __init__(self, v_in, u_in, hidden_dim, memory_size=128, attention_weight=1.0):
        super().__init__()
        self.ext_attention = ExternalAttentionLayer(v_in + u_in, hidden_dim, u_in, memory_size)
        # 注意力使用比例：1.0表示完全使用注意力（原始行为），0.0-1.0之间可以调整注意力影响
        # 通过缩放注意力输出来控制其影响程度
        self.attention_weight = attention_weight
        # 如果attention_weight < 1.0，需要添加一个投影层来将原始输入映射到hidden_dim
        # 注意：combined_input的维度是动态的，需要在forward中延迟初始化投影层
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
        # attention_weight=1.0: 完全使用注意力输出（原始行为）
        # attention_weight=0.5: 注意力输出和原始输入投影各占50%
        # attention_weight=0.0: 完全跳过注意力，只使用原始输入投影
        if self.attention_weight >= 1.0:
            # 完全使用注意力输出（原始行为，保持向后兼容）
            out = attn_out
        else:
            # 需要混合注意力输出和原始输入投影
            # 延迟初始化投影层（因为combined_dim在运行时才知道）
            if self.input_projection is None or self._input_proj_dim != combined_dim:
                self._input_proj_dim = combined_dim
                self.input_projection = nn.Linear(combined_dim, attn_out.size(1)).to(combined_input.device)
            
            original_proj = self.input_projection(combined_input)
            
            if self.attention_weight <= 0.0:
                # 完全跳过注意力，只使用原始输入投影
                out = original_proj
            else:
                # 加权混合：attention_weight * attn_out + (1 - attention_weight) * 原始输入投影
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
        # 确保所有参数都是 tensor 类型，而不是 tuple
        # 如果是 tuple，取第一个元素；如果仍然是 tuple，递归解包
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
            
        # 确保所有参数都是 tensor 类型
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
        return self.global_mlp(out).to(device)  # 添加 .to(device)


class GHGEAT(nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim, attention_weight=1.0):
        super(GHGEAT, self).__init__()
        # attention_weight: 注意力使用比例，范围[0.0, 1.0]
        # 1.0表示完全使用注意力（原始行为），0.0表示完全跳过注意力
        # 可以通过调整此参数来控制注意力机制的影响程度
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

        # MLP for A
        self.mlp1a = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mlp2a = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3a = nn.Linear(hidden_dim, 1)

        # MLP for B
        self.mlp1b = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mlp2b = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3b = nn.Linear(hidden_dim, 1)
        
        # 🔑 初始化输出层权重：使用较小的初始化范围，避免初始预测值过大
        # 这有助于防止训练初期的数值不稳定
        self._initialize_output_layers()

    def _initialize_output_layers(self):
        """初始化输出层权重，使用较小的范围以避免数值不稳定"""
        # 对输出层mlp3a和mlp3b使用较小的权重初始化
        # 使用均匀分布初始化，范围在[-0.01, 0.01]之间
        # 这样可以确保初始预测值不会太大
        for layer in [self.mlp3a, self.mlp3b]:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.uniform_(layer.weight, -0.01, 0.01)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.uniform_(layer.bias, -0.01, 0.01)

    def generate_sys_graph(self, x, edge_attr, batch_size, n_mols=2):
        src = np.arange(batch_size)
        dst = np.arange(batch_size, n_mols * batch_size)
        self_connection = np.arange(n_mols * batch_size)
        one_way = np.concatenate((src, dst, self_connection))
        other_way = np.concatenate((dst, src, self_connection))
        edge_index = torch.tensor([list(one_way), list(other_way)], dtype=torch.long).to(device)  # 添加 .to(device)
        sys_graph = Data(x=x.to(device), edge_index=edge_index, edge_attr=edge_attr.to(device))  # 添加 .to(device)
        return sys_graph

    def forward(self, solvent, solute, T):
        # Molecular descriptors based on MOSCED model
        ap1 = solvent.ap.reshape(-1, 1).to(device)  # 添加 .to(device)
        ap2 = solute.ap.reshape(-1, 1).to(device)  # 添加 .to(device)
        bp1 = solvent.bp.reshape(-1, 1).to(device)  # 添加 .to(device)
        bp2 = solute.bp.reshape(-1, 1).to(device)  # 添加 .to(device)
        topopsa1 = solvent.topopsa.reshape(-1, 1).to(device)  # 添加 .to(device)
        topopsa2 = solute.topopsa.reshape(-1, 1).to(device)  # 添加 .to(device)
        intra_hb1 = solvent.inter_hb.to(device)  # 添加 .to(device)
        intra_hb2 = solute.inter_hb.to(device)  # 添加 .to(device)

        u1 = torch.cat((ap1, bp1, topopsa1), axis=1).to(device)  # 添加 .to(device)
        u2 = torch.cat((ap2, bp2, topopsa2), axis=1).to(device)  # 添加 .to(device)

        single_node_batch = False
        if solvent.edge_attr.shape[0] == 0 or solute.edge_attr.shape[0] == 0:
            solvent = cat_dummy_graph(solvent)
            solute = cat_dummy_graph(solute)
            u1_dummy = torch.tensor([1, 1, 1]).reshape(1, -1).to(device)  # 添加 .to(device)
            u2_dummy = torch.tensor([1, 1, 1]).reshape(1, -1).to(device)  # 添加 .to(device)
            u1 = torch.cat((u1, u1_dummy), axis=0)
            u2 = torch.cat((u2, u2_dummy), axis=0)
            single_node_batch = True

        # Solvent GraphNet
        solvent.x = solvent.x.to(device)  # 添加 .to(device)
        solvent.edge_index = solvent.edge_index.to(device)  # 添加 .to(device)
        solvent.edge_attr = solvent.edge_attr.to(device)  # 添加 .to(device)
        solvent.batch = solvent.batch.to(device)  # 添加 .to(device)

        solute.x = solute.x.to(device)  # 添加 .to(device)
        solute.edge_index = solute.edge_index.to(device)  # 添加 .to(device)
        solute.edge_attr = solute.edge_attr.to(device)  # 添加 .to(device)
        solute.batch = solute.batch.to(device)  # 添加 .to(device)

        x1, edge_attr1, u1 = self.graphnet1(solvent.x,
                                            solvent.edge_index,
                                            solvent.edge_attr,
                                            u1,
                                            solvent.batch)
        # 确保返回值是 tensor 而不是 tuple
        x1 = ensure_tensor(x1)
        edge_attr1 = ensure_tensor(edge_attr1)
        u1 = ensure_tensor(u1)
        
        x1 = self.gnorm1(x1, solvent.batch)
        x1, edge_attr1, u1 = self.graphnet2(x1,
                                            solvent.edge_index,
                                            edge_attr1,
                                            u1,
                                            solvent.batch)
        # 再次确保返回值是 tensor
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
        # 确保返回值是 tensor
        x2 = ensure_tensor(x2)
        edge_attr2 = ensure_tensor(edge_attr2)
        u2 = ensure_tensor(u2)
        
        x2 = self.gnorm1(x2, solute.batch)
        x2, edge_attr2, u2 = self.graphnet2(x2,
                                            solute.edge_index,
                                            edge_attr2,
                                            u2,
                                            solute.batch)
        # 再次确保返回值是 tensor
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

        T = T.x.reshape(-1, 1) + 273.15
        # 基本的温度保护：确保温度不会过小（防止除以0）
        # 摄氏度转开尔文：T_C + 273.15，正常范围应该在200K-600K
        # 只对极端异常值进行保护，使用一个非常小的阈值（如10K）
        T = torch.clamp(T, min=10.0)

        A = F.gelu(self.mlp1a(xg))
        A = F.gelu(self.mlp2a(A))
        A = self.mlp3a(A)

        B = F.gelu(self.mlp1b(xg))
        B = F.gelu(self.mlp2b(B))
        B = self.mlp3b(B)
        
        # 🔑 数值稳定性保护：防止B值过大导致B/T溢出
        # 限制B的范围，避免B/T产生inf（例如，如果T=273K，B>1e38会导致溢出）
        # 使用合理的物理范围：通常B的值应该在-1e4到1e4之间
        B_max = 1e4  # 防止B/T溢出的最大B值
        B_min = -1e4  # 防止B/T溢出的最小B值
        B = torch.clamp(B, min=B_min, max=B_max)
        
        # 🔑 计算B/T时添加额外保护：如果B/T可能溢出，进行裁剪
        # 避免除以非常小的T导致B/T过大
        B_over_T = B / T
        
        # 检查B/T是否会导致数值溢出（如inf或nan）
        # 如果T太小或B太大，B/T可能非常大，需要裁剪
        B_over_T_max = 1e6  # B/T的最大合理值
        B_over_T_min = -1e6  # B/T的最小合理值
        B_over_T = torch.clamp(B_over_T, min=B_over_T_min, max=B_over_T_max)
        
        output = A + B_over_T
        
        # 🔑 最终输出保护：确保输出不包含inf或nan
        # 如果输出包含异常值，进行裁剪
        output_max = 1e6  # 输出的最大合理值
        output_min = -1e6  # 输出的最小合理值
        output = torch.clamp(output, min=output_min, max=output_max)
        
        # 🔑 检测并处理NaN/Inf：如果仍有异常值，用0替换（作为最后手段）
        # 这不应该发生，但如果发生了，至少不会导致训练崩溃
        if torch.any(~torch.isfinite(output)):
            print(f"⚠️ 警告：检测到输出中的NaN/Inf，位置: {torch.where(~torch.isfinite(output))}")
            output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

        return output.to(device)  # 添加 .to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)