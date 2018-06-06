import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(567)

edges = pd.read_csv('ml-20m/ratings.csv')
del edges['timestamp']

train, test = train_test_split(edges, train_size=0.8, stratify=edges.userId)

# # --
# # Most frequent predictions

# vcs   = train[1].value_counts()
# top_k = pd.Series(vcs.head(10).index)
# pk = pd.DataFrame(test).groupby(0)[1].apply(lambda x: top_k.isin(x))
# pk.mean()

# # 0.85
# # ?? slides say it should be ~0.11 

# # --

train = train.sort_values(['userId', 'movieId']).reset_index(drop=True)
train['movieRating'] = train.movieId.astype(str)# + ',' + train.rating.astype(str)
train_feats = train.groupby('userId').movieRating.apply(lambda x: ':'.join(x))

test  = test.sort_values(['userId', 'movieId']).reset_index(drop=True)
test['movieRating'] = test.movieId.astype(str)# + ',' + test.rating.astype(str)
test_feats  = test.groupby('userId').movieRating.apply(lambda x: ':'.join(x))

assert train_feats.shape[0] == test_feats.shape[0]

train_feats.to_csv('data/train.txt', sep='\t', header=None)
test_feats.to_csv('data/test.txt', sep='\t', header=None)