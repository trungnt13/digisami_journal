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
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from pprint import pprint
import os
import unicodedata
import cPickle
from collections import OrderedDict, defaultdict

import numpy as np
from odin import fuel as F
from odin.utils import get_all_files
from odin.utils.cache_utils import cache_memory, cache_disk
from odin.preprocessing import textgrid
from odin.preprocessing.speech import compute_delta
from odin import visual
from itertools import groupby

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import (TfidfVectorizer, CountVectorizer,
                                             HashingVectorizer)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import manifold

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import re
_replace_pat = [
    re.compile('\[\.*\]'), # [...]
    re.compile('\(.*\)'), # (laugh)
    re.compile('\*.*\*'), # *laugh*
    re.compile('[a-zA-Z]*-\s?'), # wh-, th-
]

support_dataset = [
    "estonian",
    "finnish",
    "sami"
]
support_datatype = [
    "trans",
    "laugh",
    "topic"
]

DELIMITER = ":"


def is_overlap(s1, e1, s2, e2):
    """True: if (s2,e2) is sufficient overlap on (s1,e1)"""
    if s2 >= s1 and e2 <= e1: # winthin (s1, e1)
        return True
    elif (s2 < s1 and e2 > s1) and (s1 - s2) < (e2 - s1): # s1 within (s2, e2)
        return True
    elif (s2 < e1 and e2 > e1) and (e1 - s2) > (e2 - e1): # e1 within (s2, e2)
        return True
    return False


def preprocess_text(text):
    for r in _replace_pat:
        text = r.sub('', text)
    return text


def to_discrete_time(timestamp, unit):
    if unit is not None and unit > 0:
        return int(float(timestamp) / unit)
    return float(timestamp)


# ===========================================================================
# Tokenizer
# ===========================================================================
def get_wordnet_pos(treebank_tag):
    # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


@cache_disk
def tokenize(docs, keep_stopword=True, keep_oov=False, keep_punct=True,
             lemmatizer=False, stem=False, lower=False):
    from odin.utils import flatten_list
    # ====== prepare ====== #
    docs = flatten_list(docs)
    # ====== process ====== #
    tk_docs = []
    lemmatiser = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for i, doc in enumerate(docs):
        tokens_pos = nltk.pos_tag(word_tokenize(doc)) # Generate list of tokens
        tokens = []
        for word, pos in tokens_pos:
            if not keep_stopword and word in stopwords.words('english'):
                continue
            if not keep_punct and pos in (',', '.', ":", "(", ")", "[", "]",
                                          '\'', '\"'):
                continue
            if lemmatizer:
                pos = get_wordnet_pos(pos)
                if len(pos) > 0:
                    word = lemmatiser.lemmatize(word, pos)
            if stem:
                word = stemmer.stem(word)
            if lower:
                word = word.lower()
            tokens.append(word)
        tk_docs.append(tokens)
    return tk_docs


@cache_disk
def get_annotation(path, dataset, datatype, language='eng',
                   conversation=True, discrete_time=0.01):
    """
    path: str
        path to downloaded digisami project datasets
    dataset: str
        sami, estonian, finnish
    datatype: str
        trans - transcription, laugh - laughter annotations,
        topic - annotation.
    conversation: bool
        collapse adjacent responses of the same speaker into one
        response.
    discrete_time: float or None
        convert time to discrete units, if NOne, keep original time

    Return
    ------
    dictionary: {
        fname: [(spkID, start, end, text), ...],
        fname*: [xmin, xmax],
        header: "..."
    }
    """
    if dataset not in support_dataset:
        raise ValueError("dataset must be: %s" % str(support_dataset))
    if datatype not in support_datatype:
        raise ValueError("datatype must be: %s" % str(support_datatype))
    if datatype == support_datatype[0] and language is None:
        raise ValueError("you must specify 'language' if datatype='trans'.")
    if language is not None and language not in ('eng', 'fin', 'est', 'sami'):
        raise ValueError("Support languages include: 'eng', 'fin', 'est', 'sami'.")
    path = os.path.join(path, dataset, '%s_csv') % datatype
    if not os.path.exists(path):
        raise RuntimeError("Cannot find annotations at path: %s" % path)
    files = get_all_files(path, filter_func=lambda x: '.csv' == x[-4:])
    # ====== extrct annotations into a dictionary ====== #
    anno = {}
    for fpath in files:
        fname = os.path.basename(fpath).split('.')[0]
        with open(fpath, 'r') as f:
            xmin = float(f.readline().split(DELIMITER)[1])
            xmax = float(f.readline().split(DELIMITER)[1])
            anno[fname + '*'] = (xmin, xmax)
            anno['header'] = f.readline().split(DELIMITER)
            data = [i.replace('\n', '').split(':') for i in f]
            # decode utf8
            data = [d[:-3] + [to_discrete_time(d[-3], discrete_time), # start time
                              to_discrete_time(d[-2], discrete_time), # end time
                              preprocess_text(d[-1].decode('utf8'))] # text
                for d in data]
            data = [d for d in data if len(d[-1]) > 0]
            # transcriptions
            if datatype == support_datatype[0]:
                data = [d[1:] for d in data if d[0] == language]
                # collapse into conversation
                if conversation:
                    final_data = []; curr_spk = -1; curr_start = 0
                    curr_end = 0; curr_text = ''
                    for spk, start, end, text in data:
                        # new spk found
                        if spk != curr_spk:
                            if curr_spk > 0:
                                curr_text = curr_text.strip()
                                curr_text = [i[0] for i in groupby(curr_text.split(' '))]
                                curr_text = (' '.join(curr_text)).strip()
                                final_data.append(
                                    (curr_spk, curr_start, curr_end, curr_text))
                            (curr_spk, curr_start, curr_end,
                                curr_text) = spk, start, end, text
                        else:
                            curr_end = end
                            curr_text += ' ' + text
                    data = final_data
            # assign data to result
            anno[fname] = data
    return anno


@cache_disk
def get_boundingbox(path, dataset,
                    person_box=False, head_box=True,
                    body_box=True, legs_box=False,
                    discrete_time=0.01):
    """
    Return
    ------
    dictionary: {
        fname: [X_left, X_right],
        fname*: [xmin, xmax],
    }
    where, "X" is np.ndarray of shape (nb_samples, start-end-boundingbox)
    order of bouning box: person-head-body-legs
    """
    # ===========================================================================
    # person, head, body
    # Load data
    # The format of the data in the CSV files is as follows:
    # Column 1:        video frame number
    # Column 2-5:      left person bounding box
    # Column 6-9:      right person bounding box
    # Column 10-13:    left head bounding box
    # Column 14-17:    right head bounding box
    # Column 18-21:    left body bounding box
    # Column 22-25:    right body bounding box
    # Column 26-29:    left legs bounding box
    # Column 30-33:    right legs bounding box
    # Each bounding box consists of 4 numbers (x, y, w, h) as follows:
    # x:    x-coordinate of top left corner of bounding box (distance from left edge of frame, in pixels)
    # y:    y-coordinate of top left corner of bounding box (distance from top edge of frame, in pixels)
    # w:   width of bounding box (in pixels)
    # h:   height of bounding box (in pixels)
    # ===========================================================================
    # load ds path
    dspath = os.path.join(path, dataset)
    # bounding box path
    boundingbox_path = os.path.join(dspath, 'boundingbox')
    if not os.path.exists(boundingbox_path):
        raise RuntimeError("Cannot find bounding box folder at path: %s" %
            boundingbox_path)
    box_files = {os.path.basename(f)[:-4]: f for f in
    get_all_files(boundingbox_path, filter_func=lambda x: '.csv' == x[-4:])}
    box = {} # return
    # ====== get xmin xmax ====== #
    trans = get_annotation(path, dataset, 'trans')
    for name in trans.keys():
        if '*' in name:
            xmin, xmax = trans[name]
            # read the bounding box
            X = np.genfromtxt(box_files[name[:-1]], delimiter=',',
                    dtype='int32')[:, 1:] # ignore first index
            if discrete_time is not None:
                n = int((xmax - xmin) / discrete_time)
                timestamp = np.linspace(0, n, X.shape[0], dtype='int32')
            else:
                timestamp = np.linspace(xmin, xmax, X.shape[0])
            timestamp = np.array([(start, end)
                for start, end in zip(timestamp, timestamp[1:])])
            X = X[:-1] # ignore the last frame
            assert len(X) == len(timestamp)
            # order: person, head, body, legs
            X_left = []; X_right = []
            if person_box: # person
                X_left.append(X[:, 0:4])
                X_right.append(X[:, 4:8])
            if head_box: # head
                X_left.append(X[:, 8:12])
                X_right.append(X[:, 12:16])
            if body_box:
                X_left.append(X[:, 16:20])
                X_right.append(X[:, 20:24])
            if legs_box: # include legs bounding box
                X_left.append(X[:, 24:28])
                X_right.append(X[:, 28:32])
            X_left = np.concatenate([timestamp] + X_left, axis=-1)
            X_right = np.concatenate([timestamp] + X_right, axis=-1)
            # assign the result
            box[name] = (xmin, xmax)
            box[name[:-1]] = (X_left, X_right)
    # return
    return box


def topic_visualization(path, dataset, discrete_time=0.01,
                        save_path=None, method='cluster', nb_topics=6):
    def preprocess_word(w):
        w = w.lower()
        if w == u'acquintance':
            return u'acquaintance'
        elif w == u'c\xe1diz':
            return u'cadiz'
        return w
    # ====== extract topics ====== #
    topics = get_annotation(path, dataset, 'topic',
                            discrete_time=discrete_time)
    _ = []
    timestamp = []
    files = []
    for name in topics.keys():
        if '*' in name:
            xmin, xmax = topics[name]
            for start, end, text in topics[name[:-1]]:
                timestamp.append((start * discrete_time / xmax,
                                  end * discrete_time / xmax))
                _.append(text.encode('utf-8').decode('utf-8'))
                files.append(name[:-1])
    topics = _
    nb_top_words = 12
    # ====== pre-processing ====== #
    tk = tokenize(topics, keep_stopword=False, keep_oov=True, keep_punct=False,
                  lemmatizer=True, stem=True, lower=True)
    _1 = []; _2 = []; _3 = []
    original_topics = []
    for i, j in enumerate(tk):
        if len(j) > 0:
            _1.append(j) # tokenized topic
            _2.append(timestamp[i]) # timestamp
            _3.append(files[i])
            original_topics.append(topics[i])
    tk = _1
    timestamp = _2
    files = _3
    assert len(timestamp) == len(tk) == len(files)
    # ====== processing ====== #
    if method == 'embedding':
        embedder = F.load_glove(ndim=50)
        tk_embedd = np.array(
            [np.mean([embedder[preprocess_word(w)] for w in doc], axis=0)
             for doc in tk])
        clustering = manifold.MDS(n_components=2)
        # manifold.TSNE(n_components=2)
        tk_embedd = clustering.fit_transform(tk_embedd)
        plt.scatter(tk_embedd[:, 0], tk_embedd[:, 1])
        # visual.plot_scatter(x, y)
        visual.plot_save('/tmp/tmp.pdf', tight_plot=True)
    elif method == 'cluster' or method == 'lda':
        vectorizer = TfidfVectorizer(max_df=0.95, max_features=None,
                                     min_df=2, stop_words=None)
        tk_vec = vectorizer.fit_transform([' '.join(i) for i in tk])
        terms = vectorizer.get_feature_names()
        if method == 'cluster':
            km = KMeans(n_clusters=nb_topics, init='k-means++', max_iter=100,
                n_init=8)
            km.fit(tk_vec)
            # show topic terms
            print("Top terms per cluster:")
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()
            for i in range(nb_topics):
                print("Topic %d:" % i, end='')
                for ind in order_centroids[i, :nb_top_words]:
                    print(' %s' % terms[ind], end='')
                print()
            # prediction for each topic
            Y = []
            for topic in tk:
                topic = ' '.join(topic)
                pred = km.predict(vectorizer.transform([topic]))
                # print(topic, pred)
                Y.append(pred.tolist()[0])
        elif method == 'lda':
            lda = LatentDirichletAllocation(n_topics=nb_topics,
                                           max_iter=8,
                                           learning_method='online',
                                           batch_size=len(tk),
                                           learning_offset=50.,
                                           random_state=0)
            Y = np.argmax(lda.fit_transform(tk_vec), axis=-1)
            assert len(Y) == len(tk)
            print("\nTopics in LDA model:")
            for topic_idx, tp in enumerate(lda.components_):
                topic_terms = [terms[i]
                               for i in tp.argsort()[:-nb_top_words - 1:-1]]
                print("Topic #%d:" % topic_idx, " ".join(topic_terms))
                samples = [original_topics[i]
                    for i, y in enumerate(Y) if y == topic_idx]
                samples = np.random.choice(samples, size=min(25, len(samples)),
                                           replace=False)
                for s in samples:
                    print("  ", s)
    # ====== topic spectum ====== #
    assert len(Y) == len(timestamp) == len(tk) == len(files)
    n = nb_topics * 6
    spectra = np.zeros(shape=(nb_topics, n))
    results = defaultdict(list) # returned result
    for f, topic, pred, (start, end) in zip(files, tk, Y, timestamp):
        topic = ' '.join(topic)
        results[f].append((start, end, topic, pred))
        start = int(start * n); end = int(end * n)
        spectra[pred, start:end] += 1
    # print the spectra
    for spec in spectra.astype('int32').tolist():
        print(spec)
    # ====== plot ====== #
    spectra = spectra.astype('float') / spectra.sum(axis=1)[:, np.newaxis]
    f, axes = plt.subplots(nb_topics, sharex=True, sharey=False)
    nb_duplicate = 4
    for i, (spec, axis) in enumerate(zip(spectra, axes)):
        spec = np.vstack([spec[None, :]] * nb_duplicate)
        axis.imshow(spec, interpolation='bilinear', cmap=plt.cm.Reds)
        # xticks
        axis.set_xticks(np.linspace(0, n, 3))
        axis.set_xticklabels(['Begin', 'Middle', 'End'],
            rotation=0, fontsize=12)
        # ytick
        axis.set_yticks([nb_duplicate / 2.5])
        axis.set_yticklabels(['Topic %d' % (i + 1)], fontsize=12)
        axis.grid(False)
        # bounding box
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.suptitle(dataset.upper(), fontsize=18)
    return results


def multimodal_correlation(path, dataset,
    method='cluster', discrete_time=0.01, nb_topics=6):
    def stack(x, n=1, shift=1, delta=2):
        if n > 1:
            idx = list(range(0, x.shape[0], shift))
            _ = [x[i:i + n].ravel() for i in idx
                 if (i + n) <= x.shape[0]]
            x = np.asarray(_) if len(_) > 1 else _[0]
        # compute delta
        x = compute_delta(x, width=9, order=delta, axis=0, trim=True)
        if delta == 1:
            x = x[0]
        else:
            x = np.concatenate(x, axis=-1)
        x = np.mean(x, axis=-1)
        return x

    def plot_correlation(mat, xlabels, ylabels, title):
        plt.figure()
        mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
        axis = plt.gca()
        axis.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
        # xticks
        tick_marks = np.arange(len(xlabels))
        axis.set_xticks(tick_marks)
        axis.set_xticklabels(xlabels, rotation=90, fontsize=12)
        # yticks
        tick_marks = np.arange(len(ylabels))
        axis.set_yticks(tick_marks)
        axis.set_yticklabels(ylabels, fontsize=12)
        # Turns off grid on the left Axis.
        axis.grid(False)
        plt.suptitle(title)
    duration = {} # store duration of each file
    topics_color = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
    # ====== get bounding box ====== #
    bounding_box = get_boundingbox(path, dataset, person_box=False,
        head_box=True, body_box=True, legs_box=True, discrete_time=0.01)
    bb = {}
    for name in bounding_box.keys():
        if '*' in name:
            xmin, xmax = bounding_box[name]
            duration[name[:-1]] = int(xmax - xmin)
            left, right = bounding_box[name[:-1]]
            timestamp = left[:, :2]
            head = stack(
                np.concatenate([left[:, 2:6], right[:, 2:6]], axis=-1))
            body = stack(
                np.concatenate([left[:, 6:10], right[:, 6:10]], axis=-1))
            legs = stack(
                np.concatenate([left[:, 10:14], right[:, 10:14]], axis=-1))
            # start, end, head, body, legs
            data = np.concatenate((timestamp, head[:, None], body[:, None], legs[:, None]),
                axis=-1)
            bb[name[:-1]] = data
    # ====== laugh ====== #
    laughter = get_annotation(path, dataset, 'laugh', discrete_time=discrete_time)
    laughter = {i: j for i, j in laughter.iteritems() if '*' not in i}
    # ====== topic ====== #
    topics = topic_visualization(path, dataset, discrete_time=0.01, method=method,
                                 nb_topics=nb_topics)
    # ====== visualize the correlation ====== #
    n = 400
    # head, body, leg -> topics
    topic_movement_corr_ff = np.zeros(shape=(3, nb_topics))
    topic_movement_corr_fm = np.zeros(shape=(3, nb_topics))
    topic_movement_corr_mm = np.zeros(shape=(3, nb_topics))
    # free_laugh, speech_laugh -> topics
    topic_laugh_corr_ff = np.zeros(shape=(2, nb_topics))
    topic_laugh_corr_fm = np.zeros(shape=(2, nb_topics))
    topic_laugh_corr_mm = np.zeros(shape=(2, nb_topics))
    for name, dura in duration.iteritems():
        print("Processing:", name)
        box = bb[name]
        lau = [i[1:] for i in laughter[name]]
        tpi = [(int(start * dura / discrete_time), int(end * dura / discrete_time), topic, pred)
            for start, end, topic, pred in sorted(topics[name], key=lambda x: x[0])]
        # timeunit, perform subsampling
        timestamp = np.linspace(0, dura / discrete_time, n)
        # bounding box
        head = []; body = []; legs = []
        for start, end in zip(timestamp, timestamp[1:]):
            _h = []; _b = []; _l = []
            for x in box:
                s = x[0]
                if s >= end: break
                elif start <= s < end:
                    _h.append(x[2])
                    _b.append(x[3])
                    _l.append(x[4])
            head.append(np.mean(_h))
            body.append(np.mean(_b))
            legs.append(np.mean(_l))
        # laughter
        laugh_info = []
        for start, end in zip(timestamp, timestamp[1:]):
            _ = []
            for s, e, l in lau:
                if start <= s < end:
                    _.append(l[:2])
            laugh_info.append(_)
        # topic
        pred = []
        for start, end in zip(timestamp, timestamp[1:]):
            _ = -1
            for s, e, tp, pd in tpi:
                if s <= start < e:
                    _ = pd
            pred.append(_)
        # store the correlation
        assert len(pred) == len(laugh_info) == len(head) == len(body) == len(legs)
        for pr, la, he, bo, le in zip(pred, laugh_info, head, body, legs):
            if '_FF_' in name:
                move_corr = topic_movement_corr_ff
                laugh_corr = topic_laugh_corr_ff
            elif '_MM_' in name:
                move_corr = topic_movement_corr_mm
                laugh_corr = topic_laugh_corr_mm
            else:
                move_corr = topic_movement_corr_fm
                laugh_corr = topic_laugh_corr_fm
            move_corr[0, pr] += np.abs(he)
            move_corr[1, pr] += np.abs(bo)
            move_corr[2, pr] += np.abs(le)
            if len(la) > 0:
                for laugh_type in la:
                    if laugh_type == u'fl':
                        laugh_corr[0, pr] += 1
                    else:
                        laugh_corr[1, pr] += 1
        # plot
        if True:
            # movements
            f, axes = plt.subplots(5, sharex=True, sharey=False)
            axes[0].plot(head); axes[0].set_yticks([]); axes[0].set_ylabel('head')
            axes[1].plot(body); axes[1].set_yticks([]); axes[1].set_ylabel('body')
            axes[2].plot(legs); axes[2].set_yticks([]); axes[2].set_ylabel('legs')
            # laughter
            for i, j in enumerate(laugh_info):
                if len(j) > 0:
                    axes[3].axvline(x=i, ymin=0, ymax=1, color='r', linewidth=1.,
                                    alpha=0.6)
            axes[3].set_yticks([]); axes[3].set_ylabel('laugh')
            # topic
            for i, j in enumerate(pred):
                if j >= 0: axes[4].axvline(x=i, ymin=0, ymax=1, color=topics_color[j],
                                 linewidth=2.5, alpha=0.8)
            axes[4].set_yticks([]); axes[4].set_ylabel('topic')
            # xticks
            axes[4].set_xticks(np.linspace(0, n - 1, 3))
            axes[4].set_xticklabels(['Begin', 'Middle', 'End'],
                rotation=0, fontsize=12)
            # final adjust
            f.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            plt.suptitle(name, fontsize=18)
    # ====== correlation matrix ====== #
    plot_correlation(topic_movement_corr_ff,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['head', 'body', 'legs'],
        title="Topic - Movement correlation (FF)")
    plot_correlation(topic_movement_corr_fm,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['head', 'body', 'legs'],
        title="Topic - Movement correlation (FM)")
    plot_correlation(topic_movement_corr_mm,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['head', 'body', 'legs'],
        title="Topic - Movement correlation (MM)")
    plot_correlation(
        topic_movement_corr_mm + topic_movement_corr_ff + topic_movement_corr_fm,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['head', 'body', 'legs'],
        title="Topic - Movement correlation")

    plot_correlation(topic_laugh_corr_mm,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['Free Laugh', 'Speech Laugh'],
        title="Topic - Laughter correlation (MM)")
    plot_correlation(topic_laugh_corr_ff,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['Free Laugh', 'Speech Laugh'],
        title="Topic - Laughter correlation (FF)")
    plot_correlation(topic_laugh_corr_fm,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['Free Laugh', 'Speech Laugh'],
        title="Topic - Laughter correlation (FM)")
    plot_correlation(
        topic_laugh_corr_mm + topic_laugh_corr_fm + topic_laugh_corr_ff,
        ['Topic %d' % (i + 1) for i in range(nb_topics)],
        ['Free Laugh', 'Speech Laugh'],
        title="Topic - Laughter correlation")
    visual.plot_save("/Users/trungnt13/tmp/tmp.pdf")
