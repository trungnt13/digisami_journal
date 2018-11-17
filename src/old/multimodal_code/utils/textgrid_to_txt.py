from __future__ import print_function, division, absolute_import

import os
from odin.utils import get_all_files
from odin.preprocessing import textgrid


def save(S, path):
    f = open(path, 'w')
    S = sorted(S, key=lambda x: x[1])
    for s, t, text in S:
        f.write(('S1' if s else 'S2') + ':' + text.encode('utf8') + '\n')
    f.close()


path = "/Users/trungnt13/OneDrive - University of Helsinki/data/DigiSami_ConversationalSpeech/transcriptions"
outpath = "/Users/trungnt13/OneDrive - University of Helsinki/data/DigiSami_ConversationalSpeech/transcriptions_txt"
files = get_all_files(path, filter_func=lambda x: 'TextGrid' in x)
for f in files:
    name = os.path.basename(f).replace('TextGrid', '')
    S_fin = []
    S_sam = []
    print(f)
    with open(f, 'r') as f:
        t = textgrid.TextGrid(f)
        for i in t:
            print(i.tier_name())
            is_fin = True if 'suomennos' in i.tier_name() else False
            is_s1 = True if 'A' in i.tier_name() else False
            is_speech = 'lauseet' in i.tier_name() or 'suomennos' in i.tier_name()
            if is_speech:
                for j in i:
                    if len(j[-1]) > 0:
                        S = S_fin if is_fin else S_sam
                        S.append((is_s1, j[0], j[-1]))
    # english
    save(S_fin, os.path.join(outpath, name) + '[fin].txt')
    # estonia
    save(S_sam, os.path.join(outpath, name) + '[sam].txt')
