
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
import numpy as np

from GOOD.networks.models.GINs import GINFeatExtractor
from GOOD.networks.models.GINvirtualnode import vGINFeatExtractor
from vector_quantize_pytorch import VectorQuantize
from models.gnnconv import GNN_node


class RbfHSIC(nn.Module):
    def __init__(self, sigma=None):
        super(RbfHSIC, self).__init__()
        self.sigma = sigma

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X.matmul(X.t())
        X_sqnorms = torch.diag(XX)
        r2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        r2 = r2.clamp(min=0)
        if sigma is None:
            try:
                sigma = torch.median(r2[r2 > 0])
            except Exception:
                sigma = 1.0
            sigma = sigma.detach()
        return torch.exp(-r2 / (2 * sigma + 1e-8))

    def forward(self, X, Y):
        n = X.size(0)
        if n > 5000:
            perm = torch.randperm(n)[:5000]
            X = X[perm]
            Y = Y[perm]
            n = 5000

        Kx = self._kernel(X, self.sigma)
        Ky = self._kernel(Y, self.sigma)

        # Centering Matrix H
        H = torch.eye(n, device=X.device) - (1. / n) * torch.ones((n, n), device=X.device)

        # HSIC = Tr(Kx * H * Ky * H) / (n-1)^2
        KxH = torch.mm(Kx, H)
        KyH = torch.mm(Ky, H)

        hsic_value = torch.trace(torch.mm(KxH, KyH)) / ((n - 1) ** 2)
        return hsic_value

class Separator(nn.Module):
    def __init__(self, args, config):
        super(Separator, self).__init__()
        if args.dataset.startswith('GOOD'):
            if config.model.model_name == 'GIN':
                self.r_gnn = GINFeatExtractor(config, without_readout=True)
            else:
                self.r_gnn = vGINFeatExtractor(config, without_readout=True)
            emb_d = config.model.dim_hidden
        else:
            self.r_gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                                  drop_ratio=args.dropout, gnn_type=args.gnn_type)
            emb_d = args.emb_dim

        self.separator = nn.Sequential(nn.Linear(emb_d, emb_d * 2),
                                       nn.BatchNorm1d(emb_d * 2),
                                       nn.ReLU(),
                                       nn.Linear(emb_d * 2, emb_d),
                                       nn.Sigmoid())
        self.args = args

    def forward(self, data):
        if self.args.dataset.startswith('GOOD'):
            node_feat = self.r_gnn(data=data)
        else:
            node_feat = self.r_gnn(data)
        score = self.separator(node_feat)
        pos_score_on_node = score.mean(1)
        pos_score_on_batch = scatter_add(pos_score_on_node, data.batch, dim=0)
        neg_score_on_batch = scatter_add((1 - pos_score_on_node), data.batch, dim=0)
        return score, pos_score_on_batch + 1e-8, neg_score_on_batch + 1e-8


class DiscreteEncoderAdvanced(nn.Module):
    def __init__(self, args, config):
        super(DiscreteEncoderAdvanced, self).__init__()
        self.args = args
        self.config = config

        self.register_buffer('alpha', torch.tensor(args.alpha_init))

        if args.dataset.startswith('GOOD'):
            emb_dim = config.model.dim_hidden
            if config.model.model_name == 'GIN':
                self.gnn = GINFeatExtractor(config, without_readout=True)
            else:
                self.gnn = vGINFeatExtractor(config, without_readout=True)
            self.classifier = nn.Sequential(*([nn.Linear(emb_dim, config.dataset.num_classes)]))
        else:
            emb_dim = args.emb_dim
            self.gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                                drop_ratio=args.dropout, gnn_type=args.gnn_type)
            self.classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                            nn.BatchNorm1d(emb_dim * 2), nn.ReLU(),
                                            nn.Dropout(), nn.Linear(emb_dim * 2, 1))

        self.pool = global_mean_pool

        self.vq_c = VectorQuantize(dim=emb_dim, codebook_size=args.num_ec, commitment_weight=args.commitment_weight,
                                   decay=0.9)
        self.vq_s = VectorQuantize(dim=emb_dim, codebook_size=args.num_es, commitment_weight=args.commitment_weight,
                                   decay=0.9)

    def forward(self, data, score):
        if self.args.dataset.startswith('GOOD'):
            node_feat = self.gnn(data=data)
        else:
            node_feat = self.gnn(data)

        H_for_VQc = node_feat * score + self.alpha * node_feat * (1 - score)
        H_for_VQs = node_feat * (1 - score) + self.alpha * node_feat * score

        node_v_feat_c, _, cmt_loss_c = self.vq_c(H_for_VQc.unsqueeze(0))
        node_v_feat_s, _, cmt_loss_s = self.vq_s(H_for_VQs.unsqueeze(0))

        node_v_feat_c = node_v_feat_c.squeeze(0)
        node_v_feat_s = node_v_feat_s.squeeze(0)

        # 最终表征组合
        c_node_feat = (node_feat * score) + node_v_feat_c
        s_node_feat = (node_feat * (1 - score)) + node_v_feat_s

        c_graph_feat = self.pool(c_node_feat, data.batch)
        s_graph_feat = self.pool(s_node_feat, data.batch)
        c_logit = self.classifier(c_graph_feat)

        return c_logit, c_graph_feat, s_graph_feat, cmt_loss_c, cmt_loss_s, c_node_feat, s_node_feat


class MyModel(nn.Module):
    def __init__(self, args, config):
        super(MyModel, self).__init__()
        self.args = args
        self.config = config

        self.separator = Separator(args, config)
        self.encoder = DiscreteEncoderAdvanced(args, config)

        emb_dim = config.model.dim_hidden if args.dataset.startswith('GOOD') else args.emb_dim
        self.hsic = RbfHSIC()

        self.mix_proj = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim),
                                      nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                      nn.Dropout(), nn.Linear(emb_dim, emb_dim))

    def forward(self, data):
        score, pos_score, neg_score = self.separator(data)
        c_logit, c_graph_feat, s_graph_feat, cmt_loss_c, cmt_loss_s, c_node_feat, s_node_feat = self.encoder(data,
                                                                                                             score)

        loss_reg = torch.abs(pos_score / (pos_score + neg_score) - self.args.gamma * torch.ones_like(pos_score)).mean()
        return c_logit, c_graph_feat, s_graph_feat, cmt_loss_c, cmt_loss_s, loss_reg, c_node_feat, s_node_feat

    def mix_cs_proj(self, c_f: torch.Tensor, s_f: torch.Tensor):
        perm = np.random.permutation(c_f.size(0))
        mix_f = torch.cat([c_f, s_f[perm]], dim=-1)
        return self.mix_proj(mix_f)