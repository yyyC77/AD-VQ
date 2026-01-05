# filename: 22.py (修改版)

import os
import logging
from tqdm import tqdm
from munch import Munch, munchify
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np

from GOOD import register
from GOOD.utils.config_reader import load_config
from GOOD.utils.metric import Metric
from GOOD.data.dataset_manager import read_meta_info
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.train import nan2zero_get_mask

from args_parse import args_parser

from models import MyModel
from exputils import initialize_exp, set_seed, get_dump_path, describe_model
from dataset import DrugOODDataset

logger = logging.getLogger()


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Runner:
    def __init__(self, args, writer, logger_path):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}')
        if args.dataset.startswith('GOOD'):
            cfg_path = os.path.join(args.config_path, args.dataset, args.domain, args.shift, 'base.yaml')
            cfg, _, _ = load_config(path=cfg_path)
            cfg = munchify(cfg)
            cfg.device = self.device
            dataset, meta_info = register.datasets[cfg.dataset.dataset_name].load(dataset_root=args.data_root,
                                                                                  domain=cfg.dataset.domain,
                                                                                  shift=cfg.dataset.shift_type,
                                                                                  generate=cfg.dataset.generate)
            read_meta_info(meta_info, cfg)
            cfg.model.dropout_rate = args.dropout
            cfg.train.train_bs = args.bs
            cfg.random_seed = args.random_seed
            loader = register.dataloader[cfg.dataset.dataloader_name].setup(dataset, cfg)
            self.train_loader, self.valid_loader, self.test_loader = loader['train'], loader['val'], loader['test']
            self.metric = Metric()
            self.metric.set_score_func(dataset['metric'] if type(dataset) is dict else getattr(dataset, 'metric'))
            self.metric.set_loss_func(dataset['task'] if type(dataset) is dict else getattr(dataset, 'task'))
            cfg.metric = self.metric
        else:
            dataset = DrugOODDataset(name=args.dataset, root=args.data_root)
            self.train_loader = DataLoader(dataset[dataset.train_index], batch_size=args.bs, shuffle=True,
                                           drop_last=True)
            self.valid_loader = DataLoader(dataset[dataset.valid_index], batch_size=args.bs, shuffle=False)
            self.test_loader = DataLoader(dataset[dataset.test_index], batch_size=args.bs, shuffle=False)
            self.metric = Metric()
            self.metric.set_loss_func(task_name='Binary classification')
            self.metric.set_score_func(metric_name='ROC-AUC')
            cfg = Munch({'metric': self.metric, 'model': Munch({'model_level': 'graph'})})
        self.cfg = cfg

        if args.dataset in ['GOODHIV', 'GOODPCBA'] or args.dataset.startswith('ic50') or args.dataset.startswith(
                'ec50'):
            self.metric.lower_better = -1
        elif args.dataset == 'GOODZINC':
            self.metric.lower_better = 1
        self.model = MyModel(args=args, config=self.cfg).to(self.device)
        self.opt_G = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.total_step = 0
        self.writer = writer
        describe_model(self.model, path=logger_path)
        self.logger_path = logger_path

    def run(self):
        best_valid_score = float('inf') if self.metric.lower_better == 1 else -1.0
        best_test_score = float('inf') if self.metric.lower_better == 1 else -1.0

        for e in range(self.args.epoch):
            if e < self.args.anneal_epochs:
                new_alpha_value = self.args.alpha_init * (1 - e / self.args.anneal_epochs)
                self.model.encoder.alpha = torch.tensor(new_alpha_value, device=self.device)
            else:
                self.model.encoder.alpha = torch.tensor(0.0, device=self.device)
            self.writer.add_scalar('hyper/alpha', self.model.encoder.alpha.item(), e)

            self.train_step(e)
            valid_score = self.test_step(self.valid_loader)
            logger.info(f"E={e}, valid={valid_score:.5f}, best_test_score={best_test_score:.5f}")

            if (self.metric.lower_better == 1 and valid_score < best_valid_score) or \
                    (self.metric.lower_better == -1 and valid_score > best_valid_score):
                best_valid_score = valid_score
                best_test_score = self.test_step(self.test_loader)
                logger.info(f"UPDATE E={e}, valid={valid_score:.5f}, new_test_score={best_test_score:.5f}")
        logger.info(f"Final test score: {best_test_score:.5f}")

    @torch.no_grad()
    def test_step(self, loader):
        self.model.eval()
        y_pred, y_gt = [], []
        for data in loader:
            data = data.to(self.device)
            logit, _, _, _, _, _, _, _ = self.model(data)
            mask, _ = nan2zero_get_mask(data, 'None', self.cfg)
            pred, target = eval_data_preprocess(data.y, logit, mask, self.cfg)
            y_pred.append(pred)
            y_gt.append(target)
        return eval_score(y_pred, y_gt, self.cfg)

    def train_step(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"E [{epoch}] Alpha [{self.model.encoder.alpha.item():.2f}]")
        for data in pbar:
            data = data.to(self.device)


            self.opt_G.zero_grad()

            c_logit, c_f, s_f, cmt_c, cmt_s, reg_loss, c_node, s_node = self.model(data)

            mask, target = nan2zero_get_mask(data, 'None', self.cfg)
            cls_loss = (self.metric.loss_func(c_logit, target.float(), reduction='none') * mask).sum() / mask.sum()

            mix_f = self.model.mix_cs_proj(c_f, s_f)
            inv_loss = -(F.normalize(c_f.detach(), dim=1) * F.normalize(mix_f, dim=1)).sum(dim=1).mean()

            hsic_loss = self.model.hsic(c_node, s_node)
            info_loss = hsic_loss
            cmt_loss = self.args.cmt_w1 * cmt_c + self.args.cmt_w2 * cmt_s
            loss_G = cls_loss + self.args.inv_w * inv_loss + self.args.info_w * info_loss + cmt_loss + self.args.reg_w * reg_loss
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt_G.step()

            # pbar.set_postfix_str(f"G_Loss={loss_G.item():.4f}, D_Loss={loss_D.item():.4f}")
            pbar.set_postfix_str(f"Loss={loss_G.item():.4f}, HSIC={hsic_loss.item():.4f}")

            self.writer.add_scalar('loss/G_total', loss_G.item(), self.total_step)
            self.writer.add_scalar('loss/HSIC', hsic_loss.item(), self.total_step)
            self.total_step += 1


def main():
    args = args_parser()

    torch.cuda.set_device(int(args.gpu))
    logger = initialize_exp(args)
    set_seed(args.random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    runner = Runner(args, writer, logger_path)
    runner.run()
    writer.close()


if __name__ == '__main__':
    main()