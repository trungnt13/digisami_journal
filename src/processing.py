from __future__ import print_function, division, absolute_import

import os
from collections import defaultdict
from six.moves import cPickle

import numpy as np
from scipy import stats
from odin.utils import (flatten_list, cache_disk, one_hot,
                        as_list, ctext, UnitTimer, Progbar,
                        cpu_count)
from odin import fuel as F
from odin.stats import train_valid_test_split, freqcount

from const import featpath
from topic_clustering import train_topic_clusters


GENDER = [None, 'F', 'M']
LAUGH = [None, 'fl', 'st']
ALL_LAUGH = [
    None,
    'fl, b', 'fl, e', 'fl, m', 'fl, o', 'fl, p',
    'st, b', 'st, e', 'st, m', 'st, o', 'st, p'
]
# extract all text once for all
ds = F.Dataset(featpath, read_only=True)
with open(os.path.join(ds.path, 'topic'), 'rb') as f:
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

    def process(self, name, X):
        lang = name.split('/')[0]
        laugh = self.laugh[name]
        gender = self.gender[lang]
        # ====== create labels ====== #
        laugh_feat = np.zeros(shape=(X[0].shape[0],), dtype='float32')
        gen_feat = np.zeros(shape=(X[0].shape[0],), dtype='float32')
        for s, e, anno, spkID in laugh:
            if anno in ALL_LAUGH:
                # laugh annotation
                if self.mode == 'bin':
                    lau = 1
                elif self.mode == 'tri':
                    lau = LAUGH.index(anno.split(',')[0])
                elif self.mode == 'all':
                    lau = ALL_LAUGH.index(anno)
                else:
                    raise RuntimeError("Unknown `mode`='%s'" % self.mode)
                laugh_feat[s:e] = lau
                # gender features
                gen = GENDER.index(gender[spkID])
                gen_feat[s:e] = gen
        # ====== add new labels and features ====== #
        X.append(one_hot(gen_feat, nb_classes=len(GENDER)))
        X.append(one_hot(laugh_feat, nb_classes=self.nb_classes))
        return name, X

    def shape_transform(self, shapes):
        ref_shp, ref_ids = shapes[0]
        n = ref_shp[0]
        # Gender
        shapes.append(
            ((n, len(GENDER)), list(ref_ids)))
        # laugh
        shapes.append(
            ((n, self.nb_classes), list(ref_ids)))
        return shapes


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

    def process(self, name, X):
        topic = self.topic[name]
        # ====== topic labels ====== #
        labels = np.full(shape=(X[0].shape[0],),
                         fill_value=self.nb_topics,
                         dtype='float32')
        for s, e, anno in topic:
            # get info
            topicID = self.model[anno]
            # set info
            labels[s:e] = topicID
        # ====== add new features ====== #
        X.append(one_hot(labels, self.nb_topics + 1))
        return name, X

    def shape_transform(self, shapes):
        """
        Parameters
        ----------
        shapes: list of shape
            list of shape tuple
        indices: dict
            {name: nb_samples}
        """
        ref_shp, ref_ids = shapes[0]
        n = ref_shp[0]
        shapes.append(
            ((n, self.nb_topics + 1), list(ref_ids)))
        return shapes


# ===========================================================================
# Main
# ===========================================================================
def get_dataset(dsname=['est'],
                feats=['mspec', 'pitch', 'sap'],
                mode='bin',
                context=30, step=1, seq=True,
                nb_topics=6, unite_topics=False,
                ncpu=None, seed=12082518,
                return_single_data=False):
    """
    dsname: str (est, fin, sam)
    feats: list of str
    normalize: list of str, name of all need-to-be-normalized features
    gender: if True include gender to the labels
    mode:
        'bin'- binary laugh and non-laugh
        'tri' - speech laugh
        'all' - all type of laugh
    unite_topics: bool,
        if True, train 1 topic clustering model for all dataset
    return_single_data: bool
        if True, don't split train, valid, test and return single
        Feeder of all utterances

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
        * Topic features (n, 1)
        * Gender features (n, 1)
        * Laugh features (n, 1)
    """
    # ====== prepare arguments ====== #
    np.random.seed(seed)
    dsname = as_list(dsname, t=str)
    feats = [s.lower() for s in as_list(feats, t=str)]
    mode = str(mode)
    ds = F.Dataset(featpath, read_only=True)
    context = int(context)
    step = int(step)
    if ncpu is None:
        ncpu = cpu_count() - 1
    nb_classes = get_nb_classes(mode)
    print(ctext("#Classes:", 'cyan'), nb_classes)
    # ====== annotations ====== #
    with open(os.path.join(ds.path, 'laugh'), 'rb') as f:
        laugh = cPickle.load(f)
    with open(os.path.join(ds.path, 'topic'), 'rb') as f:
        topic = cPickle.load(f)
    with open(os.path.join(ds.path, 'gender'), 'rb') as f:
        gender = cPickle.load(f)
    # ====== get the indices of given languages ====== #
    indices = [(name, start, end)
               for name, (start, end) in ds['indices'].items()
               if any(i + '/' in name for i in dsname)]
    print(ctext("#Utterances:", 'cyan'), len(indices))
    print(ctext("#Laugh:", 'cyan'), len(laugh))
    print(ctext("#Topic:", 'cyan'), len(topic))
    # ====== get all types of data ====== #
    data = [ds[i] for i in feats]
    # ====== recipes ====== #
    recipes = [
        # Adding topic features
        TopicTrans(topic, nb_topics=nb_topics, unite_topics=unite_topics),
        # Laugh annotation and gender feature
        LaughTrans(laugh, gender=gender, mode=mode),
        # Sequencing of Stacking
        F.recipes.Sequencing(frame_length=context,
                             step_length=step, end='cut',
                             label_mode='last') if seq else
        F.recipes.Stacking(left_context=context // 2,
                           right_context=context // 2,
                           shift=step)
    ]
    # ====== split train test ====== #
    if not return_single_data:
        train, valid, test = train_valid_test_split(indices,
            # stratified split by language cluster
            cluster_func=lambda x: x[0].split('/')[0],
            # split by utterances
            idfunc=lambda x: x[0].split('/')[1].split('.')[0],
            train=0.6, inc_test=True, seed=np.random.randint(10e8))
        assert len(train) + len(test) + len(valid) == len(indices)
        print(ctext("#Train Utterances:", 'cyan'), len(train),
              freqcount(train, key=lambda x: x[0].split('/')[0]))
        print(ctext("#Valid Utterances:", 'cyan'), len(valid),
              freqcount(valid, key=lambda x: x[0].split('/')[0]))
        print(ctext("#Test Utterances:", 'cyan'), len(test),
             freqcount(test, key=lambda x: x[0].split('/')[0]))
        # ====== create feeder and recipes ====== #
        train = F.Feeder(F.DataDescriptor(data=data, indices=train),
                         batch_mode='batch', ncpu=ncpu, buffer_size=18)
        valid = F.Feeder(F.DataDescriptor(data=data, indices=valid),
                         batch_mode='batch', ncpu=max(1, ncpu // 2), buffer_size=4)
        test = F.Feeder(F.DataDescriptor(data=data, indices=test),
                        batch_mode='file', ncpu=ncpu, buffer_size=1)
        train.set_recipes(recipes)
        valid.set_recipes(recipes)
        test.set_recipes(recipes).set_batch(batch_size=256, seed=None)
        print(ctext("Train shape:", 'cyan'), train.shape)
        print(ctext("Valid shape:", 'cyan'), valid.shape)
        print(ctext("Test shape:", 'cyan'), test.shape)
    else:
        train = F.Feeder(F.DataDescriptor(data=data, indices=indices),
                         batch_mode='batch', ncpu=ncpu, buffer_size=18)
        train.set_recipes(recipes)
    print(train)
    # ====== some test ====== #
    # prog = Progbar(target=len(train), print_summary=True, print_report=True)
    # for x in train:
    #     prog['Data'] = str([i.shape for i in x])
    #     prog.add(x[0].shape[0])
    # ====== estimate nb_classes ====== #
    if return_single_data:
        return train, nb_classes
    return train, valid, test, nb_classes


def get_pca(feat_name):
    pass
