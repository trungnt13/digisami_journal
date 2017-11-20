from __future__ import print_function, division, absolute_import

inpath = [
    "/mnt/sdb1/digisami_data/estonian",
    "/mnt/sdb1/digisami_data/finnish",
    "/mnt/sdb1/digisami_data/sami_conv"
]

segpath = '/home/trung/data/sami_segs'
featpath = "/home/trung/data/sami_feat"


CUT_DURATION = 30
FRAME_LENGTH = 0.025
STEP_LENGTH = 0.005


def utt_id(name):
    """ Return unique utterance ID for given segment"""
    lang, name = name.split('/')
    name = name.split('.')[0]
    if lang == 'sam':
        _ = name.split('_')
        name = _[0] + '_' + _[-1].split('-')[0]
    if lang == 'fin':
        name = name[:3]
    if lang == 'est':
        name = name[:4]
    return lang + '/' + name
