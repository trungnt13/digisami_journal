from __future__ import print_function, division, absolute_import

import numpy as np
from processing import get_dataset

train, valid, test, nb_classes = get_dataset(
    dsname=['est', 'fin', 'sam'],
    feats=['mspec', 'pitch', 'vad'],
    normalize=['mspec'],
    mode='tri',
    context=100, hop=20, seq=True,
    nb_topics=6, unite_topics=True,
    ncpu=4, seed=12)
for X, y in train:
    print(X.shape, y.shape)
print(train.shape)
print(valid.shape)
print(test.shape)
