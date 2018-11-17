from __future__ import print_function, division, absolute_import

import os
from shutil import copyfile
from collections import defaultdict

import numpy as np
import numba as nb

from odin import fuel as F
from odin import preprocessing as pp
from odin.utils import bidict
from utils import laugh_labels

win = 0.015
shift = 0.005
n_filters = 40
n_ceps = 13
ncpu = 12
fs = None

# ===========================================================================
# Path
# audio_path, ds16_path, ds32_path, laughter_path
# ===========================================================================
estonia = (
    '/mnt/sdb1/digisami/EstonianFirstEncounters/audio',
    '/home/trung/data/estonia_audio',
    '/mnt/sdb1/digisami/EstonianFirstEncounters/laughter'
)

finnish = (
    '/mnt/sdb1/digisami/FinnishFirstEncounters/audio',
    '/home/trung/data/finnish_audio',
    '/mnt/sdb1/digisami/FinnishFirstEncounters/laughter',
)

sami = (
    '/mnt/sdb1/digisami/DigiSami_ConversationalSpeech/audio',
    '/home/trung/data/sami_audio',
    '/mnt/sdb1/digisami/DigiSami_ConversationalSpeech/laughter',
)


# ===========================================================================
# Helper
# ===========================================================================
@nb.jit('i4(f8, f8[:], f8)', nopython=True, cache=True)
def check_timestamp_in_given_ranges(timestamp, ranges, win):
    midpoint = timestamp + win / 2
    for i in range(0, len(ranges), 3):
        start = ranges[i]
        end = ranges[i + 1]
        anno = ranges[i + 2]
        # midpoint in the range, laughing detected
        if midpoint >= start and midpoint <= end:
            return anno
    return 0


class LaughReader(object):
    """ LaughReader
    replace: str
        textgrid file is in following format: [name]_laughter.TextGrid
        the extension "_laughter.TextGrid" will be replace by this string
        to match the name of your features
    """

    def __init__(self, path, win, shift):
        super(LaughReader, self).__init__()
        self.path = path
        self.win = win
        self.shift = shift
        self.laugh_dict = {}
        self.laugh_duration = defaultdict(list)
        # ====== init ====== #
        self._which_data = None
        if 'Estonian' in path:
            self.__init_estionia()
            self._which_data = 'estonia'
        elif 'DigiSami' in path:
            self.__init_sami()
            self._which_data = 'sami'
        elif 'Finnish' in path:
            self.__init_finnish()
            self._which_data = 'finnish'
        else:
            raise Exception()

    def __init_finnish(self):
        # ====== start processing ====== #
        for p in os.listdir(self.path):
            if '.TextGrid' not in p:
                continue
            name = p[:3]
            if name in self.laugh_dict:
                raise Exception('Duplicated file name.')
            print('Processing Finnish:', name)
            # get the TextGrid
            p = os.path.join(self.path, p)
            trans = pp.textgrid.TextGrid(p)
            for i in trans:
                tmp = []
                if 'laughter' in i.tier_name():
                    for start, end, anno in i:
                        start = float(start); end = float(end)
                        anno = anno.replace(u'st, b ', u'st, b')
                        self.laugh_duration[anno].append(end - start)
                        tmp.append((start, end, laugh_labels[anno]))
                    self.laugh_dict[name] = np.array(tmp).ravel()

    def __init_sami(self):
        # ====== start processing ====== #
        for p in os.listdir(self.path):
            if '.TextGrid' not in p:
                continue
            name = p.replace('_laughter.TextGrid', '.wav')
            if name in self.laugh_dict:
                raise Exception('Duplicated file name.')
            print('Processing Sami:', name)
            # get the TextGrid
            p = os.path.join(self.path, p)
            trans = pp.textgrid.TextGrid(p)
            for i in trans:
                tmp = []
                if 'laugh' in i.tier_name():
                    for start, end, anno in i:
                        start = float(start); end = float(end)
                        anno = anno.replace(u'st, b ', u'st, b')
                        self.laugh_duration[anno].append(end - start)
                        tmp.append((start, end, laugh_labels[anno]))
                    self.laugh_dict[name] = np.array(tmp).ravel()

    def __init_estionia(self):
        # create transcription
        for p in os.listdir(self.path):
            p = os.path.join(self.path, p)
            if '.TextGrid' in p:
                name = os.path.basename(p).replace('_laughter.TextGrid', '_audio.wav')
                print('Processing Estionia:', name)
                # get laugh annotations
                for i in pp.textgrid.TextGrid(p):
                    if 'S1Laugh' in i.tier_name():
                        s1minmax = i.min_max()
                        laugh1 = []
                        for start, end, anno in i:
                            start = float(start)
                            end = float(end)
                            if len(anno) > 0:
                                self.laugh_duration[anno].append(end - start)
                                laugh1.append((start, end, laugh_labels[anno]))
                        laugh1 = np.asarray(laugh1).ravel()
                    if 'S2Laugh' in i.tier_name():
                        s2minmax = i.min_max()
                        laugh2 = []
                        for start, end, anno in i:
                            start = float(start)
                            end = float(end)
                            if len(anno) > 0:
                                self.laugh_duration[anno].append(end - start)
                                laugh2.append((start, end, laugh_labels[anno]))
                        laugh2 = np.asarray(laugh2).ravel()
                # save the output
                if s1minmax != s2minmax:
                    raise Exception('Minmax Error:' + name)
                # ====== save distribution of laugh duration ====== #
                self.laugh_dict[name] = (laugh1, laugh2)

    def process(self, save_path, indices_path):
        print('Start processing ...')
        # ====== load indices ====== #
        data_dict = {}
        for i, start, end in np.genfromtxt(indices_path, dtype=str, delimiter=' '):
            data_dict[i] = int(end) - int(start)
        # ====== output path ====== #
        if os.path.exists(save_path):
            os.remove(save_path)
        transcription = F.MmapDict(save_path)
        # ====== processing ====== #
        for name, n in data_dict.iteritems():
            print(name)
            # ====== for Estonia dataset ====== #
            if self._which_data == 'estonia':
                merge_laughter = True
                laugh1, laugh2 = self.laugh_dict[name]
                # do the annotation
                laugh1 = [laugh_labels[check_timestamp_in_given_ranges(
                                       _, laugh1, self.win)]
                         for _ in np.arange(1, n + 1) * self.shift]
                laugh2 = [laugh_labels[check_timestamp_in_given_ranges(
                                       _, laugh2, self.win)]
                         for _ in np.arange(1, n + 1) * self.shift]
                if merge_laughter:
                    laugh = [i if len(j) == 0 else j
                             for i, j in zip(laugh1, laugh2)]
                else:
                    laugh = (laugh1, laugh2)
            # ====== Sami dataset ====== #
            elif self._which_data == 'sami':
                laugh = self.laugh_dict[name]
                laugh = [laugh_labels[check_timestamp_in_given_ranges(_, laugh, self.win)]
                         for _ in np.arange(1, n + 1) * self.shift]
            # ====== finnish dataset ====== #
            elif self._which_data == 'finnish':
                laugh_name = [i for i in self.laugh_dict.keys() if i in name]
                if len(laugh_name) == 0:
                    continue
                elif len(laugh_name) > 1:
                    raise Exception('Found duplicated files:' + str(laugh_name))
                laugh = self.laugh_dict[laugh_name[0]]
                laugh = [laugh_labels[check_timestamp_in_given_ranges(_, laugh, self.win)]
                         for _ in np.arange(1, n + 1) * self.shift]
            # ====== store the labels ====== #
            transcription[name] = laugh
        # save the transcription
        transcription.flush()
        transcription.close()
        # ====== test some statistic ====== #
        trans = F.MmapDict(save_path)
        stat = defaultdict(int)
        for i in trans.itervalues():
            if isinstance(i[0], (tuple, list)):
                i = i[0] + i[1]
            for j in i:
                stat[j] += 1
        for i, j in stat.iteritems():
            print(str(i), ':', j)


# ===========================================================================
# Extract audio
# ===========================================================================
def extract_audio():
    for path, out16, laugh in [estonia, finnish, sami]:
        sp = F.SpeechProcessor(segments=path, output_path=out16,
            audio_ext='.wav', fs=fs, win=win, shift=shift,
            n_filters=n_filters, n_ceps=n_ceps,
            delta_order=2, energy=True, pitch_threshold=0.5,
            get_spec=True, get_mspec=True, get_mfcc=True, get_vad=True,
            dtype='float16', ncache=0.05, ncpu=ncpu)
        sp.run()
        # ====== extract laugh labels ====== #
        reader = LaughReader(laugh, win=win, shift=shift)
        laughter_path_output = os.path.join(out16, 'laugh.dict')
        reader.process(laughter_path_output,
                       os.path.join(out16, 'indices.csv'))
        print()

extract_audio()
