#!/usr/bin/env python

"""
    rec.py
"""

import sys
import json
import argparse
import numpy as np
from time import time
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet, HPSchedule
from basenet.helpers import to_numpy, set_seeds
from basenet.text.data import SortishSampler

from torch.utils.data import Dataset, DataLoader

# --
# Helpers

class RaggedAutoencoderDataset(Dataset):
    def __init__(self, X, n_toks):
        self.X = [torch.LongTensor(xx) for xx in X]
        self.n_toks = n_toks
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.zeros((n_toks,))
        y[x] += 1
        return self.X[idx], y
    
    def __len__(self):
        return len(self.X)


def pad_collate_fn(batch, pad_value=0):
    X, y = zip(*batch)
    
    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]
    
    X = torch.stack(X, dim=-1).t().contiguous()
    y = torch.stack(y, dim=0)
    return X, y


class DestinyModel(BaseNet):
    def __init__(self, n_toks, emb_dim, weights=None):
        
        def _loss_fn(x, y):
            return F.binary_cross_entropy_with_logits(x, y)
        
        super().__init__(loss_fn=_loss_fn)
        
        self.emb = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        
        # >>
        # self.wemb = nn.Embedding(n_toks, 1, padding_idx=0)
        # self.wemb.weight.data = torch.Tensor(weights).cuda().unsqueeze(-1)
        # <<
        
        self.emb_bias   = nn.Parameter(torch.zeros(emb_dim))
        self.bn         = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_toks)
        
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        self.emb.weight.data[0] = 0
        
        torch.nn.init.normal_(self.classifier.weight.data, 0, 0.01)
        self.classifier.bias.data.zero_()
        self.classifier.bias.data -= 10
    
    def forward(self, x):
        x = self.emb(x) * self.wemb(x)
        x = x.sum(dim=1) + self.emb_bias
        x = F.relu(x)
        # x = self.bn(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        return x


class DestinyModel2(BaseNet):
    def __init__(self, n_toks, emb_dim, weights=None):
        
        def _loss_fn(x, y):
            return F.binary_cross_entropy_with_logits(x, y)
        
        super().__init__(loss_fn=_loss_fn)
        
        self.emb = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        # >>
        # self.wemb = nn.Embedding(n_toks, 1, padding_idx=0)
        # self.wemb.weight.data = torch.Tensor(weights).unsqueeze(-1)
        # <<
        
        self.emb_bias   = nn.Parameter(torch.zeros(emb_dim))
        self.bn1        = nn.BatchNorm1d(emb_dim)
        self.hidden     = nn.Linear(emb_dim, emb_dim)
        self.bn2        = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_toks)
        
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        self.emb.weight.data[0] = 0
        
        torch.nn.init.normal_(self.hidden.weight.data, 0, 0.01)
        self.hidden.bias.data.zero_()
        
        torch.nn.init.normal_(self.classifier.weight.data, 0, 0.01)
        self.classifier.bias.data.zero_()
        self.classifier.bias.data -= 10
    
    def forward(self, x):
        x = self.emb(x)
        x = x.sum(dim=1) + self.emb_bias
        x = self.bn1(F.relu(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.hidden(x)
        x = self.bn2(F.relu(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.classifier(x)
        return x


def precision(act, preds):
    return len(act.intersection(preds)) / preds.shape[0]

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sortish', action="store_true")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--emb-dim', type=int, default=400)
    parser.add_argument('--max-obs', type=int)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--no-verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    train_feats = np.load('train_feats.npy')
    test_feats  = np.load('test_feats.npy')
    
    if args.max_obs:
        sel = np.random.choice(len(train_feats), args.max_obs, replace=False)
        train_feats, test_feats = train_feats[sel], test_feats[sel]
    
    # Re-map tokens
    umovie      = np.unique(np.hstack(train_feats))
    lookup      = dict(zip(umovie, range(1, 1 + len(umovie))))
    n_toks      = max(lookup.values()) + 1
    train_feats = np.array([[lookup[tt] for tt in t] for t in train_feats])
    
    # >>
    # beta        = 0.5
    # cnts        = np.unique(np.hstack(train_feats), return_counts=True)[1]
    # bm25_weight = (1 - beta) * beta * cnts / cnts.mean()
    # bm25_weight = np.hstack([[0], bm25_weight])
    # <<
    
    test_feats  = np.array([set([lookup[tt] for tt in t if tt in lookup]) for t in test_feats])
    
    # Reorder for efficiency
    o = np.argsort([len(t) for t in test_feats])[::-1]
    print(o, file=sys.stderr)
    train_feats, test_feats = train_feats[o], test_feats[o]
    
    # --
    # Dataloaders
    
    if args.sortish:
        train_dataloader_kwargs = {
            "sampler" : SortishSampler(train_feats, batch_size=args.batch_size, batches_per_chunk=50),
        }
    else:
        train_dataloader_kwargs = {
            "shuffle" : True,
        }
    
    dataloaders = {
        "train" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=train_feats, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=True,
            **train_dataloader_kwargs
        ),
        "valid" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=train_feats, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )
    }
    
    if args.model == 1:
        model_class = DestinyModel
    elif args.model == 2:
        model_class = DestinyModel2
    else:
        raise Exception
        
    model = model_class(n_toks=n_toks, emb_dim=args.emb_dim, weights=None).to(torch.device('cuda'))
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    model.init_optimizer(
        opt=torch.optim.Adam,
        params=model.parameters(),
        # params =  list(model.emb.parameters()) +
        #     [model.emb_bias] +
        #     list(model.bn1.parameters()) +
        #     list(model.hidden.parameters()) +
        #     list(model.bn2.parameters()) +
        #     list(model.classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    t = time()
    for epoch in range(args.epochs):
        train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False)
        
        if epoch % args.eval_interval == 0:
            preds = model.predict(dataloaders, mode='valid')
            
            for i in range(preds.shape[0]):
                preds[i][train_feats[i]] = -1
            
            top_k = to_numpy(preds.topk(k=10, dim=-1)[1])
            
            p_at_01 = np.mean([precision(test_feats[i], top_k[i][:1]) for i in range(len(test_feats))])
            p_at_05 = np.mean([precision(test_feats[i], top_k[i][:5]) for i in range(len(test_feats))])
            p_at_10 = np.mean([precision(test_feats[i], top_k[i][:10]) for i in range(len(test_feats))])
            print(json.dumps(OrderedDict([
                ("epoch",   epoch),
                ("p_at_01", p_at_01),
                ("p_at_05", p_at_05),
                ("p_at_10", p_at_10),
                ("elapsed", time() - t),
            ])))
