from __future__ import print_function, division, absolute_import

# ===========================================================================
# Estonian and Finnish
# mpsec
# mpsec,pitch
# mpsec,f0
# mspec,energy
# mspec,vad
# mspec,energy,vad
# mspec,pitch,energy,vad
# mspec,pitch,f0,energy,vad
# ===========================================================================
efmspec0 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmspec1 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'pitch'],
    'normalize': ['mspec', 'pitch'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmspec2 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'f0'],
    'normalize': ['mspec', 'f0'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmspec3 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'energy'],
    'normalize': ['mspec', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmspec4 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'vad'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmspec5 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'energy', 'vad'],
    'normalize': ['mspec', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmspec6 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'pitch', 'energy', 'vad'],
    'normalize': ['mspec', 'pitch', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmspec7 = {
    'dsname': ['est', 'fin'],
    'feats': ['mspec', 'pitch', 'f0', 'energy', 'vad'],
    'normalize': ['mspec', 'pitch', 'f0', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

# ===========================================================================
# MFCC
# ===========================================================================
efmfcc0 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmfcc1 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc', 'pitch'],
    'normalize': ['mfcc', 'pitch'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmfcc2 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc', 'f0'],
    'normalize': ['mfcc', 'f0'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmfcc3 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc', 'energy'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmfcc4 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc', 'vad'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmfcc5 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc', 'energy', 'vad'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmfcc6 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc', 'pitch', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efmfcc7 = {
    'dsname': ['est', 'fin'],
    'feats': ['mfcc', 'pitch', 'f0', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'f0', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

# ===========================================================================
# Including Sami dataset
# ===========================================================================
efsmspec0 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmspec1 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'pitch'],
    'normalize': ['mspec', 'pitch'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmspec2 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'f0'],
    'normalize': ['mspec', 'f0'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmspec3 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'energy'],
    'normalize': ['mspec', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmspec4 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'vad'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmspec5 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'energy', 'vad'],
    'normalize': ['mspec', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmspec6 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'pitch', 'energy', 'vad'],
    'normalize': ['mspec', 'pitch', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmspec7 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec', 'pitch', 'f0', 'energy', 'vad'],
    'normalize': ['mspec', 'pitch', 'f0', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

# ===========================================================================
# MFCC
# ===========================================================================
efsmfcc0 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmfcc1 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch'],
    'normalize': ['mfcc', 'pitch'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmfcc2 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'f0'],
    'normalize': ['mfcc', 'f0'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmfcc3 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'energy'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmfcc4 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'vad'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmfcc5 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'energy', 'vad'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmfcc6 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efsmfcc7 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch', 'f0', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'f0', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': True,
}
