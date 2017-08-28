from __future__ import print_function, division, absolute_import

import os
from collections import defaultdict
from six.moves import cPickle

import numpy as np
from scipy import stats
from odin.utils import (flatten_list, cache_disk, one_hot,
                        as_list, ctext, UnitTimer, Progbar)
from odin import fuel as F
from odin.stats import train_valid_test_split, freqcount

from const import outpath
from topic_clustering import train_topic_clusters


GENDER = [None, 'F', 'M']
LAUGH = [None, 'fl', 'st']
ALL_LAUGH = [
    None,
    'fl, b',
    'fl, d',
    'fl, e',
    'fl, m',
    'fl, o',
    'fl, o, p',
    'fl, p',
    'st, b',
    'st, e',
    'st, m',
    'st, o',
    'st, p'
]
_INSPECT_MODE = False

# extract all text once for all
ds = F.Dataset(outpath, read_only=True)
with open(os.path.join(ds.path, 'topic'), 'r') as f:
    _ = cPickle.load(f)
    _ALL_TEXT = [j[-1] for i in _.values() for j in i]
print(ctext("Total amount of transcription:", 'red'), len(_ALL_TEXT))


# ===========================================================================
# Helper function
# ===========================================================================
def get_nb_classes(mode):
    if mode == 'bin':
        n = 2
    elif mode == 'tri':
        n = 3
    elif mode == 'all':
        n = len(ALL_LAUGH)
    else:
        raise ValueError(str(mode))
    return n


def set_inspect_mode(mode):
    global _INSPECT_MODE
    _INSPECT_MODE = mode


# ===========================================================================
# Helper class
# ===========================================================================
class LaughTrans(F.recipes.FeederRecipe):
    """
    laugh: dictionary, mapping "file_name" -> list of (start, end, laugh_anno)
    length: dictionary mapping "file_name" -> file length in second
    mode: 'bin'- binary laugh and non-laugh; 'tri' - speech laugh, free
    laugh and non-laugh; 'all' - all kind of laugh
    """

    def __init__(self, laugh, gender, mode='bin'):
        super(LaughTrans, self).__init__()
        self.laugh = laugh
        self.gender = gender
        mode = str(mode).lower()
        if mode not in ("bin", "tri", "all"):
            raise ValueError("mode must be: 'bin', 'tri', 'all' but given %s" % mode)
        self.mode = mode
        self.nb_classes = get_nb_classes(mode)

    def process(self, name, X, y):
        orig_name = str(name)
        name, start, end = name.split(':')
        lang = name.split('/')[0]
        start = float(start)
        end = float(end)
        unit = (end - start) / X[0].shape[0]
        start = int(np.floor(start / unit))
        end = int(np.floor(end / unit))
        laugh = self.laugh[name]
        gender = self.gender[lang]
        # ====== create labels ====== #
        labels = np.zeros(shape=(X[0].shape[0],), dtype='float32')
        genfeat = np.zeros(shape=(X[0].shape[0],), dtype='float32')
        for spkID, s, e, anno in laugh:
            if ('fl,' in anno or 'st,' in anno) and\
            (s <= end and start <= e):
                overlap_start = max(start, s) - start
                overlap_length = min(e, end) - max(s, start)
                # laugh annotation
                lau = 1 if self.mode == 'bin' else \
                    (LAUGH.index(anno.split(',')[0]) if self.mode == 'tri' else
                     ALL_LAUGH.index(anno))
                labels[overlap_start:overlap_start + overlap_length] = lau
                # gender features
                gen = GENDER.index(gender[spkID])
                genfeat[overlap_start:overlap_start + overlap_length] = gen
        # ====== add new labels and features ====== #
        if not _INSPECT_MODE:
            X = [np.concatenate([np.expand_dims(x, -1) if x.ndim == 1 else x
                                 for x in X], axis=-1),
                 np.expand_dims(genfeat, axis=-1)]
            y.append(one_hot(labels, self.nb_classes))
        else:
            X.append(genfeat)
            y.append(labels)
        return orig_name, X, y

    def shape_transform(self, shapes, indices):
        """
        Parameters
        ----------
        shapes: list of shape
            list of shape tuple
        indices: dict
            {name: nb_samples}
        """
        n = shapes[0][0]
        if _INSPECT_MODE:
            shapes.append((n,))
        else:
            shapes = [(n, sum(1 if len(s) == 1 else s[-1] for s in shapes)),
                      (n, 1)]
        return shapes, indices


class TopicTrans(F.recipes.FeederRecipe):

    def __init__(self, topic, nb_topics=6, unite_topics=False):
        super(TopicTrans, self).__init__()
        self.topic = topic
        self.nb_topics = nb_topics
        # ====== train topic modeling ====== #
        if unite_topics:
            text = list(_ALL_TEXT)
        else:
            text = [j[-1] for i in topic.values() for j in i]
        self.model = train_topic_clusters(text, nb_topics=nb_topics,
                                          print_log=True)

    def process(self, name, X, y):
        orig_name = name
        name, start, end = name.split(':')
        lang = name.split('/')[0]
        start = float(start)
        end = float(end)
        unit = (end - start) / X[0].shape[0]
        start = int(np.floor(start / unit))
        end = int(np.floor(end / unit))
        # ====== read topic for differnt dataset ====== #
        if lang in ('est', 'fin'):
            topic = self.topic[name]
        else:
            short_name = '_'.join(name.split('_')[:-1])
            topic = self.topic[short_name]
        # ====== topic labels ====== #
        labels = np.full(shape=(X[0].shape[0], 1),
                         fill_value=self.nb_topics,
                         dtype='float32')
        for s, e, anno in topic:
            if s <= end and start <= e:
                # get info
                topicID = self.model[anno]
                # set info
                overlap_start = max(start, s) - start
                overlap_length = min(e, end) - max(s, start)
                labels[overlap_start:overlap_start + overlap_length] = topicID
        # ====== add new features ====== #
        X.append(labels.ravel() if _INSPECT_MODE else labels)
        return orig_name, X, y

    def shape_transform(self, shapes, indices):
        """
        Parameters
        ----------
        shapes: list of shape
            list of shape tuple
        indices: dict
            {name: nb_samples}
        """
        n = shapes[0][0]
        shapes.append((n, 1))
        return shapes, indices


# ===========================================================================
# Main
# ===========================================================================
def get_dataset(dsname=['est'],
                feats=['mspec', 'pitch', 'vad'],
                normalize=['mspec'],
                mode='bin',
                context=30, hop=1, seq=True,
                nb_topics=6, unite_topics=False,
                ncpu=4, seed=12082518):
    """
    dsname: str (est, fin, sam)
    feats: list of str
    normalize: list of str, name of all need-to-be-normalized features
    gender: if True include gender to the labels
    mode:
        'bin'- binary laugh and non-laugh
        'tri' - speech laugh
        'all' - all type of laugh
    unite_topics: bool, if True, train 1 topic clustering model for all
    dataset

    Features
    --------
        * spec
        * mspec
        * mfcc
        * energy
        * pitch
        * f0
        * vad

    Note
    ----
    Order of the features:
        * all_original_features_concatenated (n, ...)
        * Gender features (n, 1)
        * Topic features (n, 1)
    """
    # ====== prepare arguments ====== #
    np.random.seed(seed)
    dsname = as_list(dsname, t=str)
    feats = [s.lower() for s in as_list(feats, t=str)]
    normalize = as_list(normalize, t=str)
    mode = str(mode)
    ds = F.Dataset(outpath, read_only=True)
    context = int(context)
    hop = int(hop)
    # ====== annotations ====== #
    with open(os.path.join(ds.path, 'laugh'), 'r') as f:
        laugh = cPickle.load(f)
    with open(os.path.join(ds.path, 'topic'), 'r') as f:
        topic = cPickle.load(f)
    with open(os.path.join(ds.path, 'gender'), 'r') as f:
        gender = cPickle.load(f)
    # ====== get the indices of given languages ====== #
    indices = [(name, start, end)
               for name, (start, end) in ds['indices'].iteritems()
               if any(i + '/' in name for i in dsname)]
    print(ctext("#Utterances:", 'cyan'), len(indices))
    # ====== get length ====== #
    length = defaultdict(lambda: [0, 0])
    for name, n1, n2 in indices:
        name, start, end = name.split(':')
        # first value is length in second
        length[name][0] = max(length[name][0], float(end))
        # second value is number of frames
        length[name][1] += n2 - n1
        # sepcial case for sami
        if 'sam/' == name[:4]:
            short_name = '_'.join(name.split('_')[:-1])
            length[short_name] = length[name]
    # length is second per frame
    length = {name: duration / n
              for name, (duration, n) in length.iteritems()}
    # convert laugh and topic to index
    laugh = {name: [(spk, int(np.floor(s / length[name])),
                     int(np.ceil(e / length[name])), l)
                    for spk, s, e, l in anno]
             for name, anno in laugh.iteritems()
             if name in length}
    topic = {name: [(int(np.floor(s / length[name])),
                     int(np.ceil(e / length[name])), tp)
                    for s, e, tp in alltopic]
             for name, alltopic in topic.iteritems()
             if name in length}
    print(ctext("#Audio Files:", 'cyan'), len(length), len(laugh), len(topic))
    # ====== INPSECTATION MODE ====== #
    if _INSPECT_MODE:
        data = [ds[i] for i in feats]
        feeder = F.Feeder(data, indices=indices, dtype='float32',
                          ncpu=ncpu, buffer_size=1)
        feeder.set_recipes([
            [F.recipes.Normalization(mean=ds[i + '_mean'],
                                     std=ds[i + '_std'],
                                     local_normalize=None,
                                     data_idx=feats.index(i))
             for i in normalize if i + '_mean' in ds],
            # Laugh annotation and gender feature
            LaughTrans(laugh, gender=gender, mode=mode),
            # Adding topic features
            TopicTrans(topic, nb_topics=nb_topics, unite_topics=unite_topics),
            F.recipes.CreateFile()
        ])
        feeder.set_batch(batch_size=2056, seed=None, shuffle_level=0)
        return feeder
    # ====== split train test ====== #
    train, valid, test = train_valid_test_split(indices,
        cluster_func=lambda x: x[0].split('/')[0],
        idfunc=lambda x: x[0].split('/')[1].split(':')[0],
        train=0.6, inc_test=True, seed=np.random.randint(10e8))
    assert len(train) + len(test) + len(valid) == len(indices)
    print(ctext("#Train Utterances:", 'cyan'), len(train),
          freqcount(train, key=lambda x: x[0].split('/')[0]))
    print(ctext("#Valid Utterances:", 'cyan'), len(valid),
          freqcount(valid, key=lambda x: x[0].split('/')[0]))
    print(ctext("#Test Utterances:", 'cyan'), len(test),
         freqcount(test, key=lambda x: x[0].split('/')[0]))
    # ====== create feeder and recipes ====== #
    data = [ds[i] for i in feats]
    train = F.Feeder(data, indices=train, dtype='float32',
                     ncpu=ncpu, buffer_size=12)
    valid = F.Feeder(data, indices=valid, dtype='float32',
                     ncpu=ncpu, buffer_size=4)
    test = F.Feeder(data, indices=test, dtype='float32',
                    ncpu=2, buffer_size=1)
    # ====== recipes ====== #
    recipes = [
        [F.recipes.Normalization(mean=ds[i + '_mean'],
                                 std=ds[i + '_std'],
                                 local_normalize=None,
                                 data_idx=feats.index(i))
         for i in normalize if i + '_mean' in ds],
        # Laugh annotation and gender feature
        LaughTrans(laugh, gender=gender, mode=mode),
        # Adding topic features
        TopicTrans(topic, nb_topics=nb_topics, unite_topics=unite_topics),
        # Sequencing of Stacking
        F.recipes.Sequencing(frame_length=context,
                             hop_length=hop,
                             end='pad', endvalue=0.,
                             endmode='post') if seq else
        F.recipes.Stacking(left_context=context // 2,
                           right_context=context // 2,
                           shift=hop)
    ]
    train.set_recipes(recipes + [F.recipes.CreateBatch()])
    valid.set_recipes(recipes + [F.recipes.CreateBatch()])
    test.set_recipes(recipes + [F.recipes.CreateFile()]
        ).set_batch(batch_size=256, seed=None)
    nb_classes = get_nb_classes(mode)
    print(ctext("Train shape:", 'cyan'), train.shape)
    print(ctext("Valid shape:", 'cyan'), valid.shape)
    print(ctext("Test shape:", 'cyan'), test.shape)
    print(ctext("#Classes:", 'cyan'), nb_classes)
    # ====== some test ====== #
    # for X in Progbar(target=train,
    #                  print_report=True, print_summary=True
    #                  ).set_iter_info(lambda x: x[0].shape[0],
    #                                  lambda x: [('X', x[0].shape),
    #                                             ('Gen', x[1].shape),
    #                                             ('Topic', x[2].shape),
    #                                             ('y', x[3].shape)]):
    #     pass
    # ====== estimate nb_classes ====== #
    return train, valid, test, nb_classes
