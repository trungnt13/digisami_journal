from __future__ import print_function, division, absolute_import

import numpy as np

######
test1 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'pitch', 'vad'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 100,
    'hop': 20,
    'seq': True,
    'nb_topics': 6,
    'unite_topics': False,
    'ncpu': 6,
    'seed': 12082518
}

test2 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'pitch', 'vad', 'gender', 'topic'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 100,
    'hop': 20,
    'seq': True,
    'nb_topics': 6,
    'unite_topics': False,
    'ncpu': 6,
    'seed': 12082518
}
