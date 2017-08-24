from __future__ import print_function, division, absolute_import

import numpy as np
from processing import get_dataset

train, valid, test = get_dataset(
    dsname=['est', 'fin', 'sam'],
    feats=['mspec', 'vad'],
    normalize=['mspec'],
    mode='bin',
    gender=False,
    context=100, hop=None, seq=True,
    nb_topics=6, unite_topics=False,
    ncpu=1, seed=12)
for X, vad, y in train:
    print(X.shape, vad.shape, y.shape)
print(train.shape)
print(valid.shape)
print(test.shape)
