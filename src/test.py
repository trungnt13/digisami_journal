from __future__ import print_function, division, absolute_import

import numpy as np
from processing import get_dataset

train, valid, test, nb_classes = get_dataset(
    dsname=['est', 'fin', 'sam'],
    feats=['mspec', 'pitch', 'vad'],
    normalize=['mspec'],
    mode='tri',
    context=100, hop=1, seq=True,
    nb_topics=6, unite_topics=True,
    ncpu=6, seed=12)
for X, gen, tpc, y in train:
    print(X.shape, gen.shape, tpc.shape, y.shape)
