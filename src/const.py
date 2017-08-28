from __future__ import print_function, division, absolute_import

import os

CUT_DURATION = 10

inpath = [
    "/mnt/sdb1/digisami_data/estonian",
    "/mnt/sdb1/digisami_data/finnish",
    "/mnt/sdb1/digisami_data/sami_conv"
]

outpath = "/home/trung/data/sami_feat"
if not os.path.exists(outpath):
    outpath = '/Volumes/backup/data/sami_feat'
