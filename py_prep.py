
# edges = pd.read_csv('data/ml-20m/ratings.csv')

# umovie = set(edges.movieId)
# lookup = dict(zip(umovie, range(len(umovie))))

# edges.movieId = edges.movieId.apply(lambda x: lookup[x])
# n_toks = int(edges.movieId.max() + 1)

# train, test = train_test_split(edges, train_size=0.8, stratify=edges.userId)

# train_feats = train.groupby('userId').movieId.apply(lambda x: sorted(list(set(x)))).values
# test_feats  = test.groupby('userId').movieId.apply(lambda x: sorted(list(set(x)))).values

# o = np.argsort([len(xx) for xx in test_feats])[::-1]
# train_feats, test_feats = train_feats[o], test_feats[o]

# np.save('train_feats', train_feats)
# np.save('test_feats', test_feats)

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