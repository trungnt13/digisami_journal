# ===========================================================================
# The output data:
# {
#   'topics' -> {'file_name': [('anno', start_time, end_time), ...], ...},
#   'transcriptions' -> ...
#   'laughter' -> ...
#   'duration' -> {'file_name': duration}
#   ...
# }
# ===========================================================================
from __future__ import print_function, division, absolute_import
import os
import unicodedata
import cPickle
from collections import OrderedDict

from odin.utils import get_all_files
from odin.preprocessing import textgrid


path = "/mnt/sdb1/digisami/EstonianFirstEncounters"
outpath = "/home/trung/data/estonia_anno"

data_types = [
    'topics', # (topic, start, end)
    'transcriptions', # (text, start, end, specker_id)
    'laughter' # (text, start, end, specker_id)
]


def name_process(name):
    name = os.path.basename(name)
    return name.replace('.TextGrid', '').replace('_topic', '').replace('_laughter', '')


def to_unicode(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    # s = unicode(s.encode('utf-8'), 'utf-8')
    return s


name_list = []
data = {i: {} for i in data_types}
data['duration'] = {}

for dt in data_types:
    for f in get_all_files(os.path.join(path, dt),
                           filter_func = lambda x: '.TextGrid' in x):
        name = name_process(f)
        name_list.append(name)
        print(name, dt)
        with open(f, 'r') as f:
            grid = textgrid.TextGrid(f)
            if name not in data['duration']:
                data['duration'][name] = grid.xmax
            else:
                data['duration'][name] = max(grid.xmax, data['duration'][name])
            # ====== read the topic ====== #
            if dt == data_types[0]:
                for i in grid:
                    if 'topic' in i.tier_name():
                        dat = [(to_unicode(topic), float(start), float(end))
                               for start, end, topic in i if len(topic) > 0]
            # ====== read the trans ====== #
            elif dt == data_types[1]:
                for i in grid:
                    if 'S1Eng' in i.tier_name():
                        s1 = [(to_unicode(topic), float(start), float(end), 0)
                              for start, end, topic in i
                              if len(topic) > 0]
                    elif 'S2Eng' in i.tier_name():
                        s2 = [(to_unicode(topic), float(start), float(end), 1)
                              for start, end, topic in i
                              if len(topic) > 0]
                dat = s1 + s2
                dat = sorted(dat, key=lambda x: x[1])
            # ====== read the laughter ====== #
            elif dt == data_types[2]:
                for i in grid:
                    if 'S1Laugh' in i.tier_name():
                        s1 = [(to_unicode(topic), float(start), float(end), 0)
                              for start, end, topic in i
                              if len(topic) > 0]
                    elif 'S2Laugh' in i.tier_name():
                        s2 = [(to_unicode(topic), float(start), float(end), 1)
                              for start, end, topic in i
                              if len(topic) > 0]
                dat = s1 + s2
                dat = sorted(dat, key=lambda x: x[1])
            # ====== error ====== #
            else:
                raise Exception('Something wrong:' + dt)
            data[dt][name] = dat

# ====== check valid name_list ====== #
name_list = list(set(name_list))
assert len(name_list) == len(get_all_files(os.path.join(path, dt),
                                           filter_func=lambda x: '.TextGrid' in x)), \
    'Wrong number of name detected'
print('Topics:')
for i in data['topics']['C_19_FF_19_20']:
    print('-', i)
print('Laughter:')
for i in data['laughter']['C_19_FF_19_20']:
    print('-', i)
print('Transcription:')
for i in data['transcriptions']['C_19_FF_19_20']:
    print('-', i)
print("Duration:", data['duration'])
# ====== dump data ====== #
print('Save data at:', outpath)
cPickle.dump(data, open(outpath, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
