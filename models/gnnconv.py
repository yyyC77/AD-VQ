import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GATConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree

import math


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction"""
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GATConvBlock(nn.Module):
    def __init__(self, emb_dim, edge_feature_dim, heads=4):
        super(GATConvBlock, self).__init__()
        self.bond_encoder = MLP(num_layers=1, input_dim=edge_feature_dim, output_dim=emb_dim, hidden_dim=emb_dim)
        self.conv = GATConv(in_channels=emb_dim, out_channels=emb_dim,
                            heads=heads, concat=False,
                            edge_dim=emb_dim, add_self_loops=False)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.conv(x, edge_index, edge_attr=edge_embedding)
        return out


class GINConv(MessagePassing):
    def __init__(self, emb_dim, edge_feature_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = MLP(num_layers=1, input_dim=edge_feature_dim, output_dim=emb_dim, hidden_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x +
            self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_weight=edge_weight)
        )
        return out

    def message(self, x_j, edge_attr, edge_weight=None):
        msg = F.relu(x_j + edge_attr)
        return msg if edge_weight is None else msg * edge_weight.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_feature_dim=10):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = MLP(num_layers=1, input_dim=edge_feature_dim, output_dim=emb_dim, hidden_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    def __init__(
            self, num_layer, emb_dim, input_dim, edge_feature_dim, drop_ratio=0.5,
            JK="last", residual=False, gnn_type='gin'
    ):
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.atom_encoder = MLP(input_dim=input_dim, hidden_dim=emb_dim, output_dim=emb_dim, num_layers=2)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_feature_dim=edge_feature_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_feature_dim=edge_feature_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConvBlock(emb_dim, edge_feature_dim=edge_feature_dim))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_type}')

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            batched_data = argv[0]
            x, edge_index = batched_data.x, batched_data.edge_index
            edge_attr, batch = batched_data.edge_attr, batched_data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        x = x.float()

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio, training=self.training
                )
            if self.residual:
                h += h_list[layer]
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation


class GNN_node_Virtualnode(torch.nn.Module):
    def __init__(
            self, num_layer, emb_dim, input_dim, drop_ratio=0.5,
            JK="last", residual=False, gnn_type='gin', edge_feature_dim=10
    ):
        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = MLP(input_dim=input_dim, hidden_dim=emb_dim, output_dim=emb_dim, num_layers=2)

        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_feature_dim=edge_feature_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_feature_dim=edge_feature_dim))
            # --- [修改] 增加 GAT 分支 ---
            elif gnn_type == 'gat':
                self.convs.append(GATConvBlock(emb_dim, edge_feature_dim=edge_feature_dim))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_type}')

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU()
            ))

    def forward(self, batched_data, edge_weight=None):
        x, edge_index = batched_data.x, batched_data.edge_index
        edge_attr, batch = batched_data.edge_attr, batched_data.batch

        x = x.float()

        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(
            batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            if isinstance(self.convs[layer], GATConvBlock):
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_weight=edge_weight)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio, training=self.training
                )
            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)
            if layer < self.num_layer - 1:
                virtualnode_embedding_temp = global_add_pool(
                    h_list[layer], batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp
                        ), self.drop_ratio, training=self.training
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp
                        ), self.drop_ratio, training=self.training
                    )
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation