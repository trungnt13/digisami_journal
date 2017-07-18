from __future__ import print_function, division, absolute_import

import os
import shutil
from odin.utils import get_all_files
from odin.preprocessing import textgrid

inpath = "/Volumes/backup/digisami_data/sami_read/trans_textgrid"
audiopath = "/Volumes/backup/digisami_data/sami_read/audio"
outpath = "/Volumes/backup/digisami_data/sami_read/trans_csv"

list1 = get_all_files(inpath, filter_func=lambda x: '.TextGrid' in x)
list2 = get_all_files(audiopath, filter_func=lambda x: '.wav' in x)

set1 = set([os.path.basename(i).replace('.TextGrid', '') for i in list1])
set2 = set([os.path.basename(i).replace('.wav', '') for i in list2])
print(set1 == set2)

# ====== PRint metadata ====== #
for f in list2:
    name = os.path.basename(f)
    spk = name.replace('.wav', '').split('_')[-1]
    lang = f.split('/')[-2].split('_')[-1]
    print(name, spk, lang)
print()
# ====== convert to CSV ====== #
for fin in list1:
    name = os.path.basename(fin).replace('.TextGrid', '.csv')
    print(name)
    fout = os.path.join(outpath, name)
    with open(fin, 'r') as f:
        tg = textgrid.TextGrid(f)
        xmin = tg.xmin
        xmax = tg.xmax
        data = []
        for i in tg:
            tier_data = []
            tier_name = i.tier_name()
            for start, end, text in i:
                if len(text) == 0: continue
                tier_data.append((tier_name, float(start), float(end),
                    text.strip().replace(u':', '').encode('utf-8').decode('utf-8')))
            tier_data = sorted(tier_data, key=lambda x: x[1])
            data += tier_data
    # save to CSV file
    with open(fout, 'w') as f:
        f.write("xmin:%f\n" % xmin)
        f.write("xmax:%f\n" % xmax)
        f.write('AnnotationType:StartTime:EndTime:Text\n')
        for dat in data:
            dat = [str(i) for i in dat[:-1]] + [dat[-1].encode('utf8')]
            f.write(':'.join(dat) + "\n")
