# ===========================================================================
# Different topic strategy
# ===========================================================================
from __future__ import print_function, division, absolute_import

topic2T = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
    'nb_topics': 2,
    'unite_topics': True,
}

topic4T = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
    'nb_topics': 4,
    'unite_topics': True,
}

topic6T = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
    'nb_topics': 6,
    'unite_topics': True,
}

topic10T = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
    'nb_topics': 10,
    'unite_topics': True,
}

topic2F = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
    'nb_topics': 2,
    'unite_topics': False,
}

topic4F = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
    'nb_topics': 4,
    'unite_topics': False,
}

topic10F = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
    'nb_topics': 10,
    'unite_topics': False,
}
