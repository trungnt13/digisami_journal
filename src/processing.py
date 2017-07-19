from __future__ import print_function, division, absolute_import

import os
from six.moves import cPickle

import numpy as np
from odin.utils import flatten_list, cache_disk, one_hot
from odin import fuel as F
from odin.stats import train_valid_test_split

from const import outpath
from topic_clustering import train_topic_clusters


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


def _nb_classes(mode, gender, dsname):
    if mode == 'bin':
        n = 2
    elif mode == 'tri':
        n = 3
    if dsname in ('estonian', 'finnish') and gender:
        n = n * 2 - 1
    return n

# extract all text once for all
_ALL_TEXT = []
for path in outpath:
    ds = F.Dataset(path, read_only=True)
    with open(os.path.join(ds.path, 'topic'), 'r') as f:
        _ = cPickle.load(f)
        _ALL_TEXT += [j[-1] for i in _.values() for j in i]


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

    def __init__(self, laugh, length, dsname,
                 mode='bin', gender=False,
                 merge_features=None):
        super(LaughTrans, self).__init__()
        self.laugh = laugh
        self.length = length
        self.gender = gender
        mode = str(mode).lower()
        if mode not in ("bin", "tri", "all"):
            raise ValueError("mode must be: 'bin', 'tri', or 'all' but given %s"
                % mode)
        self.mode = mode
        if merge_features is not None and not callable(merge_features):
            raise ValueError("`merge_features` can be None or callable.")
        self.merge_features = merge_features
        self.nb_classes = _nb_classes(mode, gender, dsname)

    def process(self, name, X, y):
        laugh = self.laugh[name]
        length = self.length[name]
        n = X[0].shape[0]
        # ====== create labels ====== #
        labels = np.zeros(shape=(n,), dtype='float32')
        for spkID, start, end, anno in laugh:
            start = int(np.floor(n / length * start))
            end = int(np.ceil(n / length * end))
            gender = None if not self.gender or ('F' != spkID[-1] and 'M' != spkID[-1]) \
                else spkID[-1]
            lab = _laugh2id(anno, self.mode, gender)
            labels[start:end] = lab
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
def get_dataset(dsname,
                features=['mspec', 'pitch', 'vad'],
                normalize=['mspec'],
                gender=False,
                merge_features=None,
                mode='bin',
                context=100, hop=None, seq=True,
                unite_topics=False,
                nb_topics=6,
                ncpu=4,
                seed=12082518):
    """
    dsname: str (estonian, finnish, sami_conv)
    features: list of str
    normalize: list of str, name of all need-to-be-normalized features
    gender: if True include gender to the labels
    unite_topics: bool, if True, train 1 topic clustering model for all
    dataset

    Note
    ----
    The topic features are added to the last features
    """
    ds = [i for i in outpath if str(dsname) in i]
    if len(ds) == 0:
        raise RuntimeError("Cannot find any dataset with name: %s" % dsname)
    ds = F.Dataset(ds[0], read_only=True)
    # ====== annotations ====== #
    with open(os.path.join(ds.path, 'laugh'), 'r') as f:
        laugh = cPickle.load(f)
    with open(os.path.join(ds.path, 'topic'), 'r') as f:
        topic = cPickle.load(f)
    length = {}
    for name, (start, end) in ds['indices_raw'].iteritems():
        length[name] = float(end - start) / ds['sr'][name]
    # ====== split train valid test ====== #
    indices = ds['indices'].items()
    np.random.seed(seed)
    np.random.shuffle(indices)
    train, valid, test = train_valid_test_split(indices, train=0.6,
        inc_test=True, seed=np.random.randint(10e8))
    assert len(train) + len(test) + len(valid) == len(indices)
    # ====== create feeder and recipes ====== #
    data = [ds[i] for i in features]
    train = F.Feeder(data, indices=train, dtype='float32',
                     ncpu=ncpu, buffer_size=6)
    valid = F.Feeder(data, indices=valid, dtype='float32',
                     ncpu=ncpu, buffer_size=2)
    test = F.Feeder(data, indices=test, dtype='float32',
                    ncpu=ncpu, buffer_size=2)
    # ====== recipes ====== #
    recipes = [
        [F.recipes.Normalization(mean=ds[i + '_mean'], std=ds[i + '_std'],
                                 local_normalize=None,
                                 data_idx=features.index(i))
         for i in normalize if i + '_mean' in ds],
        TopicTrans(topic, length, nb_topics=nb_topics,
            unite_topics=unite_topics),
        LaughTrans(laugh, length, dsname,
                   mode=mode, gender=gender,
                   merge_features=merge_features),
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
        ).set_batch(batch_size=1, seed=None)
    # ====== estimate nb_classes ====== #
    return train, valid, test, _nb_classes(mode, gender, dsname)
