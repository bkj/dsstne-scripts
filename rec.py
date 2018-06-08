#!/usr/bin/env python

"""
    rec.py
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet
from basenet.helpers import to_numpy

from torch.utils.data import Dataset, DataLoader
from basenet.text.data import SortishSampler
from torch.utils.data.sampler import SequentialSampler

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


def pad_collate_fn(batch, pad_value=1):
    X, y = zip(*batch)
    
    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]
    
    X = torch.stack(X, dim=-1)
    y = torch.stack(y, dim=0)
    return X, y


# --
# IO

edges = pd.read_csv('data/ml-20m/ratings.csv')

umovie = set(edges.movieId)
lookup = dict(zip(umovie, range(len(umovie))))

edges.movieId = edges.movieId.apply(lambda x: lookup[x])
n_toks = int(edges.movieId.max() + 1)

train, test = train_test_split(edges, train_size=0.8, stratify=edges.userId)

train_feats = train.groupby('userId').movieId.apply(lambda x: sorted(list(set(x)))).values
test_feats  = test.groupby('userId').movieId.apply(lambda x: sorted(list(set(x)))).values

o = np.argsort([len(xx) for xx in test_feats])[::-1]
train_feats, test_feats = train_feats[o], test_feats[o]

np.save('train_feats', train_feats)
np.save('test_feats', test_feats)

# >>

# train_feats = []
# for line in open('data/train-small.txt').read().splitlines():
#     mid, vals = line.strip().split('\t')
#     train_feats.append(list([int(xx) for xx in vals.split(':')]))

# test_feats = []
# for line in open('data/test-small.txt').read().splitlines():
#     mid, vals = line.strip().split('\t')
#     test_feats.append(list([int(xx) for xx in vals.split(':')]))

# train_feats, test_feats = np.array(train_feats), np.array(test_feats)

# # o = np.argsort([len(xx) for xx in test_feats])[::-1]
# # train_feats, test_feats = train_feats[o], test_feats[o]

# # umovie = np.unique(np.hstack([np.hstack(train_feats), np.hstack(test_feats)]))
# # lookup = dict(zip(umovie, range(len(umovie))))
# # n_toks = max(lookup.values()) + 1
# # train_feats = np.array([[lookup[tt] for tt in t] for t in train_feats])
# # test_feats  = np.array([[lookup[tt] for tt in t] for t in test_feats])

# train_df = pd.DataFrame({
#     "idx" : range(1, len(train_feats) + 1),
#     "val" : [':'.join(map(str, sorted(t))) for t in train_feats],
# })
# train_df.to_csv('small_train_mapped', sep='\t', header=None, index=None)

# test_df = pd.DataFrame({
#     "idx" : range(1, len(test_feats) + 1),
#     "val" : [':'.join(map(str, sorted(t))) for t in test_feats],
# })
# test_df.to_csv('small_test_mapped', sep='\t', header=None, index=None)

# np.save('train_feats-small', train_feats)
# np.save('test_feats-small', test_feats)

# <<

# --
# Define model

class DestinyModel(BaseNet):
    def __init__(self, n_toks, emb_dim=400):
        
        def _loss_fn(x, y):
            return F.binary_cross_entropy_with_logits(x, y)
        
        super().__init__(loss_fn=_loss_fn)
        
        self.emb        = nn.Embedding(n_toks, emb_dim)
        self.emb_bias   = nn.Parameter(torch.zeros(emb_dim))
        self.classifier = nn.Linear(emb_dim, n_toks)
        
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        torch.nn.init.normal_(self.classifier.weight.data, 0, 0.01)
        self.classifier.bias.data.zero_()
        self.classifier.bias.data -= 10
        
    def forward(self, x):
        x = self.emb(x).sum(dim=0) + self.emb_bias
        x = F.relu(x)
        x = self.classifier(x)
        return x


def precision(act, preds):
    return len(act.intersection(preds)) / preds.shape[0]


train_feats = np.load('train_feats.npy')
test_feats = np.load('test_feats.npy')

# Re-map tokens
umovie      = np.unique(np.hstack(train_feats))
lookup      = dict(zip(umovie, range(len(umovie))))
n_toks      = max(lookup.values()) + 1
train_feats = np.array([[lookup[tt] for tt in t] for t in train_feats])

test_feats  = [set([lookup[tt] for tt in t if tt in lookup]) for t in test_feats]

batch_size = 128
N = train_feats.shape[0]
dataloaders = {
    "train" : DataLoader(
        dataset=RaggedAutoencoderDataset(X=train_feats, n_toks=n_toks),
        # sampler=SortishSampler(train_feats, batch_size=batch_size),
        batch_size=batch_size,
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    ),
    "valid" : DataLoader(
        dataset=RaggedAutoencoderDataset(X=train_feats, n_toks=n_toks),
        sampler=SequentialSampler(train_feats),
        batch_size=batch_size,
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
}

model = DestinyModel(n_toks=n_toks).to(torch.device('cuda'))
model.verbose = True
print(model)

model.init_optimizer(
    opt=torch.optim.Adam,
    params=model.parameters(),
)

t = time()
for epoch in range(50):
    print('epoch=%d | elapsed=%f' % (epoch, time() - t), file=sys.stderr)
    train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False)
    
    if epoch % 5 == 0:
        preds = model.predict(dataloaders, mode='valid')
        
        for i in range(preds.shape[0]):
            preds[i][train_feats[i]] = -1
        
        top_k = to_numpy(preds.topk(k=10, dim=-1)[1])
        print(np.mean([precision(test_feats[i], top_k[i]) for i in range(len(test_feats))]))


