from __future__ import print_function, division, absolute_import

import os
from collections import defaultdict
from six.moves import cPickle

import numpy as np
from odin.utils import flatten_list, cache_disk, one_hot, as_list, ctext
from odin import fuel as F
from odin.stats import train_valid_test_split

from const import outpath
from topic_clustering import train_topic_clusters


# extract all text once for all
ds = F.Dataset(outpath, read_only=True)
with open(os.path.join(ds.path, 'topic'), 'r') as f:
    _ = cPickle.load(f)
    _ALL_TEXT = [j[-1] for i in _.values() for j in i]
print("Total amount of transcription:", len(_ALL_TEXT))


# ===========================================================================
# Helper function
# ===========================================================================
def _get_spks(name, dsname):
    if 'est' in dsname: # C_05_FF_06_05
        return name.split('_')[-2:]
    if 'fin' in dsname: # E10_B_20110512_FF_11_10
        return name.split('_')[-2:]
    if 'sami' in dsname: # 04_20140227_IS-2
        return name.split('_')[-1:]
    raise RuntimeError


def _laugh2id(name, mode, gender):
    if mode == 'bin':
        y = 1 if len(name) > 0 else 0
    elif mode == 'tri':
        y = 0 if len(name) == 0 else (1 if 'fl' == name[:2] else 2)
    else:
        raise ValueError('Unsupport mode: %s' % mode)
    # ====== include gender ====== #
    if y > 0:
        if gender == 'M':
            y = y * 2
        elif gender == 'F':
            y = y * 2 - 1
        elif gender is None:
            pass
        else:
            raise ValueError("Unsuport gender: %s" % gender)
    return y


def get_nb_classes(mode):
    if mode == 'bin':
        n = 2
    elif mode == 'tri':
        n = 3
    return n


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

    def __init__(self, laugh, mode='bin', gender=False):
        super(LaughTrans, self).__init__()
        self.laugh = laugh
        self.gender = gender
        mode = str(mode).lower()
        if mode not in ("bin", "tri", "all"):
            raise ValueError("mode must be: 'bin', 'tri', or 'all' but given %s"
                % mode)
        self.mode = mode
        self.nb_classes = get_nb_classes(mode)

    def process(self, name, X, y):
        name, start, end = name.split(':')
        start = float(start)
        end = float(end)
        unit = (end - start) / X[0].shape[0]
        start = int(np.floor(start / unit))
        end = int(np.floor(end / unit))
        laugh = self.laugh[name]
        # ====== create labels ====== #
        labels = np.zeros(shape=(X[0].shape[0],), dtype='float32')
        for spkID, start, end, anno in laugh:
            gender = None if not self.gender or ('F' != spkID[-1] and 'M' != spkID[-1]) \
                else spkID[-1]
            # lab = _laugh2id(anno, self.mode, gender)
            # labels[start:end] = lab
        exit()
        y.append(one_hot(labels, self.nb_classes))
        # ====== merge features ====== #
        if self.merge_features is not None:
            X = self.merge_features(X)
            if not isinstance(X, (tuple, list)):
                X = [X]
        return name, list(X), y

    def shape_transform(self, shapes, indices):
        """
        Parameters
        ----------
        shapes: list of shape
            list of shape tuple
        indices: dict
            {name: nb_samples}
        """
        if self.merge_features is not None:
            X = [np.empty(shape=(1,) + tuple(s[1:]))
                 for s in shapes]
            X = self.merge_features(X)
            if not isinstance(X, (tuple, list)):
                X = (X,)
            n = shapes[0][0]
            shapes = [(n,) + x.shape[1:] for x in X]
        return shapes, indices


class TopicTrans(F.recipes.FeederRecipe):

    def __init__(self, topic, length, nb_topics=6, unite_topics=False):
        super(TopicTrans, self).__init__()
        self.topic = topic
        self.length = length
        self.nb_topics = nb_topics
        # ====== train topic modeling ====== #
        if unite_topics:
            text = list(_ALL_TEXT)
        else:
            text = [j[-1] for i in topic.values() for j in i]
        self.model = train_topic_clusters(text, nb_topics=nb_topics,
            print_log=False)

    def process(self, name, X, y):
        topic = self.topic[name]
        length = self.length[name]
        n = X[0].shape[0]
        # ====== topic labels ====== #
        labels = np.full(shape=(n, 1), fill_value=self.nb_topics,
                         dtype='float32')
        for start, end, anno in topic:
            start = int(np.floor(n / length * start))
            end = int(np.ceil(n / length * end))
            lab = self.model[anno]
            labels[start:end, 0] = lab
        X.append(labels)
        return name, X, y

    def shape_transform(self, shapes, indices):
        """
        Parameters
        ----------
        shapes: list of shape
            list of shape tuple
        indices: dict
            {name: nb_samples}
        """
        shapes.append((shapes[0][0], 1))
        return shapes, indices


# ===========================================================================
# Main
# ===========================================================================
def get_dataset(dsname=['est'],
                feats=['mspec', 'pitch', 'vad'],
                normalize=['mspec'],
                mode='bin',
                context=100, hop=None, seq=True,
                nb_topics=6, unite_topics=False,
                ncpu=6, seed=12082518):
    """
    dsname: str (est, fin, sam)
    feats: list of str
    normalize: list of str, name of all need-to-be-normalized features
    gender: if True include gender to the labels
    mode:
        'bin'- binary laugh and non-laugh
        'tri' - speech laugh
    unite_topics: bool, if True, train 1 topic clustering model for all
    dataset

    Note
    ----
    The topic features are added to the last features
    """
    # ====== prepare arguments ====== #
    np.random.seed(seed)
    dsname = as_list(dsname, t=str)
    feats = as_list(feats, t=str)
    normalize = as_list(normalize, t=str)
    mode = str(mode)
    ds = F.Dataset(outpath, read_only=True)
    if hop is None:
        hop = context // 2
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
             for name, anno in laugh.iteritems()}
    topic = {name: [(int(np.floor(s / length[name])),
                     int(np.ceil(e / length[name])), tp)
                    for s, e, tp in alltopic]
             for name, alltopic in topic.iteritems()}
    print(ctext("#Audio Files:", 'cyan'), len(length))

    # ====== split train test ====== #
    train, valid, test = train_valid_test_split(indices,
        cluster_func=lambda x: x[0].split('/')[0],
        idfunc=lambda x: x[0].split('/')[1].split(':')[0],
        train=0.6, inc_test=True, seed=np.random.randint(10e8))
    assert len(train) + len(test) + len(valid) == len(indices)
    print(ctext("#Train Utterances:", 'cyan'), len(train))
    print(ctext("#Valid Utterances:", 'cyan'), len(valid))
    print(ctext("#Test Utterances:", 'cyan'), len(test))
    # ====== create feeder and recipes ====== #
    data = [ds[i] for i in feats]
    train = F.Feeder(data, indices=train, dtype='float32',
                     ncpu=ncpu, buffer_size=12)
    valid = F.Feeder(data, indices=valid, dtype='float32',
                     ncpu=ncpu, buffer_size=1)
    test = F.Feeder(data, indices=test, dtype='float32',
                    ncpu=2, buffer_size=1)
    # ====== recipes ====== #
    recipes = [
        [F.recipes.Normalization(mean=ds[i + '_mean'],
                                 std=ds[i + '_std'],
                                 local_normalize=None,
                                 data_idx=feats.index(i))
         for i in normalize if i + '_mean' in ds],
        # TopicTrans(topic, length,
        #            nb_topics=nb_topics, unite_topics=unite_topics),
        LaughTrans(laugh, mode=mode, gender=gender),
        # Sequencing of Stacking
        # F.recipes.Sequencing(frame_length=context,
        #                      hop_length=hop,
        #                      end='pad', endvalue=0.,
        #                      endmode='post') if seq else
        # F.recipes.Stacking(left_context=context // 2,
        #                    right_context=context // 2,
        #                    shift=hop)
    ]
    train.set_recipes(recipes + [F.recipes.CreateBatch()])
    valid.set_recipes(recipes + [F.recipes.CreateBatch()])
    test.set_recipes(recipes + [F.recipes.CreateFile()]
        ).set_batch(batch_size=256, seed=None)
    for X in valid:
        for x in X:
            print(x.shape)
            exit()
    # ====== estimate nb_classes ====== #
    return train, valid, test, get_nb_classes(mode)
