from __future__ import print_function, division, absolute_import

import os
import numpy as np
from odin.utils import get_script_path, get_all_files, ctext

log = os.path.join(get_script_path(), 'logs')
files = [os.path.basename(f).replace('.log', '')
         for f in get_all_files(log, filter_func=lambda x: '.log' in x)]

for x in set([f.split('-')[0] for f in files]):
    print("Model:", x)
    print("Config:")
    configs = sorted(
        set([f.split('-')[1] + '-' + f.split('-')[2]
             for f in files if f.split('-')[0] == x]))
    for c in configs:
        print('\t', c)
print("#Experiments:", len(files))
