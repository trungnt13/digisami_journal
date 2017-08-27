from __future__ import print_function, division, absolute_import

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV

from odin.utils import ctext, Progbar
from utils import (plot_laugh_prediction, plot_save,
                   plot_2D_threshold, plot_3D_threshold)
# y_pred: {name: (n, nb_classes)}
# y_true: {name: (n,)}
# gender: {name: (n,)}
# topic: {name: (n,)}


def report(y_pred, y_true, labels=None):
    print(ctext("Accuracy:", 'yellow'), accuracy_score(y_true, y_pred))
    print(ctext("F1:", 'yellow'), f1_score(y_true, y_pred, labels=labels, average='macro'))
    print(ctext("Confusion matrix:", 'yellow'))
    print(confusion_matrix(y_true, y_pred, labels=labels))


def title(s):
    print(ctext('******** ' + s + ' ********', 'red'))


def basic_report(y_pred, y_true, gender, topic):
    # ====== General performance ====== #
    print(ctext('=== Overall ===', 'cyan'))
    report(y_pred, y_true)
    # ====== Gender performance ====== #
    gender = gender.astype('int32')
    for gen_id, gen_name in zip((1, 2), ('Female', 'Male')):
        idx = np.logical_or(gender == 0, gender == gen_id)
        gen_true = y_true[idx]
        gen_pred = y_pred[idx]
        print(ctext('=== %s (%d) ===' % (gen_name, len(gen_true)), 'cyan'))
        report(gen_pred, gen_true)
    # ====== Topic performance ====== #
    topic = topic.astype('int32')
    all_topics = np.unique(topic)
    for tpc_id in all_topics:
        idx = topic == tpc_id
        tpc_true = y_true[idx]
        tpc_pred = y_pred[idx]
        print(ctext('=== Topic#%d (%d) ===' % (tpc_id, len(tpc_true)), 'cyan'))
        report(tpc_pred, tpc_true)


def fast_sort(x):
    """Fast convert
    {name1: [...], name2: [...]} => [......]
    by concatenate everything in fixed order
    """
    return np.concatenate(
        [i[1] for i in sorted(x.items(), key=lambda x: x[0])],
        axis=0)


# ===========================================================================
# Method 1
# ===========================================================================
def evaluate_general_performance(y_pred, y_true, gender, topic):
    name = y_pred.keys()
    # ====== general ====== #
    pred = np.argmax(fast_sort(y_pred), axis=-1)
    true = fast_sort(y_true)
    gen = fast_sort(gender)
    tpc = fast_sort(topic)
    assert len(pred) == len(true) == len(gen) == len(tpc)
    title("General performance (%d)" % (len(pred)))
    basic_report(pred, true, gen, tpc)
    # ====== each language ====== #
    all_languages = set(i.split('/')[0] for i in name)
    if len(all_languages) > 1:
        for lang in all_languages:
            pred = np.argmax(
                fast_sort({name: i for name, i in y_pred.iteritems()
                           if lang == name[:3]}),
                axis=-1)
            true = fast_sort({name: i for name, i in y_true.iteritems()
                              if lang == name[:3]})
            gen = fast_sort({name: i for name, i in gender.iteritems()
                             if lang == name[:3]})
            tpc = fast_sort({name: i for name, i in topic.iteritems()
                             if lang == name[:3]})
            assert len(pred) == len(true) == len(gen) == len(tpc)
            title('[' + lang + '] Performance (%d)' % len(true))
            basic_report(pred, true, gen, tpc)


# ===========================================================================
# Method 2
# ===========================================================================
def prediction_3D(pred, threshold_x, threshold_y):
    pred_x = pred[:, 1] >= threshold_x
    pred_y = pred[:, 2] >= threshold_y
    final_pred = []
    for i, (x, y) in enumerate(zip(pred_x, pred_y)):
        if x and not y:
            final_pred.append(1)
        elif not x and y:
            final_pred.append(2)
        elif (x and y) and (pred[i, 1] != pred[i, 2]):
            final_pred.append(1 if pred[i, 1] > pred[i, 2] else 2)
        else:
            final_pred.append(0)
    return np.array(final_pred, dtype='int32')


def evaluate_smooth_label(y_pred, y_true, gender, topic):
    name = y_pred.keys()
    np.random.shuffle(name)
    # ====== No Calibration ====== #
    pred = np.argmax(fast_sort(y_pred), axis=-1)
    true = fast_sort(y_true)
    gen = fast_sort(gender)
    tpc = fast_sort(topic)
    assert len(pred) == len(true) == len(gen) == len(tpc)
    title("NO calibration performance (%d)" % (len(pred)))
    basic_report(pred, true, gen, tpc)
    # ====== visualize ====== #
    # for n in name:
    #     print(ctext("Visualizing prediction file: %s" % n, 'magenta'))
    #     pred = y_pred[n]
    #     true = y_true[n]
    #     plot_laugh_prediction(pred, true, name=n)
    # ====== calibration ====== #
    samples = {}
    for n in name:
        lang = n.split('/')[0]
        if lang not in samples:
            samples[lang] = n
    samples = samples.values()
    pred = fast_sort({s: y_pred[s] for s in samples})
    true = fast_sort({s: y_true[s] for s in samples})
    # For binary
    if pred.shape[-1] == 2:
        threshold = np.linspace(0., 1., num=120)
        performance = [f1_score(true, (pred[:, -1] >= t).astype('int32'),
                                average='macro')
                       for t in threshold]
        plot_2D_threshold(threshold, performance)
    # For trinary
    elif pred.shape[-1] == 3:
        threshold = [(x, y)
                     for x in np.linspace(0., 1., num=50)
                     for y in np.linspace(0., 1., num=50)]
        performance = []
        for (x, y) in Progbar(threshold).set_iter_info(lambda x: 1):
            performance.append(f1_score(true, prediction_3D(pred, x, y),
                                        average='macro'))
        plot_3D_threshold(threshold, performance)
    # Error
    else:
        raise RuntimeError(str(pred.shape))
    # post processing
    best_idx = np.argmax(performance)
    best_threshold = threshold[best_idx]
    best_performance = performance[best_idx]
    print('\n', ctext('Best threshold/performance:', 'yellow'),
          best_threshold, best_performance)
    # ====== Calibration ====== #
    if pred.shape[-1] == 2:
        pred = (fast_sort(y_pred)[:, -1] >= best_threshold).astype('int32')
    else:
        pred = prediction_3D(fast_sort(y_pred), best_threshold[0], best_threshold[1])
    true = fast_sort(y_true)
    gen = fast_sort(gender)
    tpc = fast_sort(topic)
    assert len(pred) == len(true) == len(gen) == len(tpc)
    title("Calibration performance (%d)" % (len(pred)))
    basic_report(pred, true, gen, tpc)
    plot_save('/tmp/tmp.pdf')
    exit()
