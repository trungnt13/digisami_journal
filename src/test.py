from __future__ import print_function, division, absolute_import

import os
import numpy as np
from odin.utils import get_script_path, get_all_files, ctext

import processing

processing.get_dataset(dsname=['est', 'fin', 'sam'],
                       mode='all',
                       seq=True, ncpu=None)
