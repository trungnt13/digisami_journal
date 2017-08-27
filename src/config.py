from __future__ import print_function, division, absolute_import

######
test1 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'pitch', 'vad'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 30,
    'hop': 30,
    'seq': True,
    'nb_topics': 6,
    'unite_topics': False,
    'ncpu': 6,
    'seed': 12082518
}

test2 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'pitch', 'vad'],
    'normalize': ['mspec'],
    'mode': 'tri',
    'context': 30,
    'hop': 30,
    'seq': True,
    'nb_topics': 6,
    'unite_topics': False,
    'ncpu': 6,
    'seed': 12082518
}
