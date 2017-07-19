from __future__ import print_function, division, absolute_import

import numpy as np
from processing import get_dataset

train, valid, test = get_dataset('estonian',
    mode='tri',
    gender=True,
    merge_features=lambda x: (np.concatenate(x[:2] + [x[-1]], axis=-1), x[-2]),
    context=100, hop=None, seq=True,
    nb_topics=6,
    unite_topics=False,
    ncpu=1)
for X, vad, y in train:
    print(X.shape, vad.shape, y.shape)
print(train.shape)
print(valid.shape)
print(test.shape)
