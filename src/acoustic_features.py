from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from odin.utils import ArgController
args = ArgController(
).add('--seg', 'force re-run audio segmentation', False
).add('--debug', 'enable debugging', False
).parse()

import os
os.environ['ODIN'] = 'float32,thread=1,cpu=1,gpu=0'

import wave
import shutil
import numpy as np
from math import ceil
from six.moves import cPickle
from collections import defaultdict

import numpy as np

from odin import fuel as F, visual as V, nnet as N
from odin.utils import get_all_files, ctext
from odin import preprocessing as pp
from const import (inpath, featpath, segpath, SR,
                   CUT_DURATION, FRAME_LENGTH, STEP_LENGTH,
                   FMIN, FMAX,
                   utt_id)
# ===========================================================================
# Helper
# ===========================================================================
def array2D(x):
  if x.ndim == 1:
    x = x[None, :]
  return x

def sec2fra(second):
  # convert time in second to frame index
  return int(np.round(float(second) / STEP_LENGTH))

def get_partitions(f):
  f = wave.open(f, mode='r')
  duration = int(ceil(f.getnframes() / f.getframerate()))
  f.close()
  partitions = list(range(0, duration - CUT_DURATION // 2, CUT_DURATION)) + [duration]
  partitions = zip(partitions, partitions[1:])
  return partitions

def time_resolution(anno, segs):
  anno_new = {}
  for name, trans in anno.items():
    lang, name = name.split('/')
    name = name + '.wav'
    for seg_name, seg_start, seg_end in segs[name]:
      seg_anno = []
      seg_start = sec2fra(seg_start)
      seg_end = sec2fra(seg_end)
      # laughter
      if len(trans[0]) == 4: # laugh
        for start, end, laugh, spk in trans:
          start = max(seg_start, sec2fra(start))
          end = min(seg_end, sec2fra(end))
          if end - start > 0:
            seg_anno.append((start - seg_start,
                             end - seg_start, laugh, spk))
      # topics
      elif len(trans[0]) == 3: # topic
        for start, end, tpc in trans:
          start = max(seg_start, sec2fra(start))
          end = min(seg_end, sec2fra(end))
          if end - start > 0:
            seg_anno.append((start - seg_start,
                             end - seg_start, tpc))
      anno_new[lang + '/' + seg_name] = seg_anno
      # if len(seg_anno) == 0:
      #     print('NO %s:' % anno_type,
      #         name + ':' + seg_name + '(%d, %d)' % (seg_start, seg_end))
  return anno_new

def get_name(path, lang):
  return lang + '/' + os.path.basename(path).replace('.wav', '').replace('.csv', '')

# ===========================================================================
# Getting all segments
# ===========================================================================
segments = {}
laugh = {}
topic = {}
gender = {} # lang -> {spkID -> F or M}
for path in inpath:
  lang = 'est' if 'estonian' in path else ('fin' if 'finnish' in path else
                                           'sam')
  print("Input path:", path)
  # ====== read laugh anno ====== #
  for f in get_all_files(os.path.join(path, 'laugh_csv')):
    data = [
    (float(start), float(end), text, spkID)
        for (spkID, start, end, text) in array2D(np.genfromtxt(f,
        dtype='str', delimiter=':', skip_header=3))]
    if len(data) == 0:
      continue
    name = get_name(f, lang)
    laugh[name] = data
  # ====== read audio ====== #
  audio_path = os.path.join(path, 'audio')
  for f in get_all_files(audio_path, lambda x: '.wav' == x[-4:]):
    name = get_name(f, lang)
    if name not in laugh:
      continue
    segments[f] = name
  # ====== read topic anno ====== #
  for f in get_all_files(os.path.join(path, 'topic_csv')):
    topic_tmp = []
    with open(f, 'r') as f: # read csv file manually
      for i, line in enumerate(f):
        if i < 3:
          continue
        start, end, text = line[:-1].split(':')
        topic_tmp.append((float(start), float(end), text))
    name = get_name(f.name, lang)
    # The topic file is saved with different name from
    # audio file, this function convert audio file name
    # to matching topic file name
    # e.g. sam/02_20140226b_V-3.10.wav => sam/02_20140226b
    for i in laugh.keys():
      if name in i:
        topic[i] = topic_tmp
  # ====== update speaker gender ====== #
  if lang in ('est', 'fin'):
    gender[lang] = {spkID: spkID[-1]
                    for name, anno in laugh.items()
                    for _, _, _, spkID in anno
                    if (lang + '/') in name}
  else:
    sami_gender = np.genfromtxt(os.path.join(path, 'gender.csv'),
                                dtype=str, delimiter=' ', skip_header=1)
    gender['sam'] = {spkID: gen for spkID, gen in sami_gender}
print("Audio: %d (files)" % len(segments))
print("Laugh: %d (files)" % len(laugh))
print("Topic: %d (files)" % len(topic))
print("Gender: %d (spk)" % sum(len(i) for i in gender.values()))
lang_info = {name.split('/')[1] + '.wav': name.split('/')[0]
             for name in segments.values()}
# ===========================================================================
# Cutting the audio
# ===========================================================================
# ====== segments audio files ====== #
meta_segs = pp.speech.audio_segmenter(
    files=list(segments.keys()),
    outpath=segpath, max_duration=CUT_DURATION,
    sr_new=SR, best_resample=True, override=args.seg)
# ====== post processing ====== #
seg_info = defaultdict(list)
segments_new = {} # file_path -> name
duration = defaultdict(float) # utt_id -> duration in second
for name, origin, start, end in np.genfromtxt(meta_segs,
                                              delimiter=' ',
                                              skip_header=1,
                                              dtype=str):
  seg_info[origin].append((name, start, end))
  # ====== new segments list ====== #
  path = os.path.join(segpath, name)
  assert os.path.isfile(path)
  name = lang_info[origin] + '/' + name
  segments_new[path] = name
  duration[utt_id(name)] += float(end) - float(start)
# ====== convert the annotations ====== #
laugh = time_resolution(laugh, seg_info)
topic = time_resolution(topic, seg_info)
# only keep segments that contains laugh events
valid_name = {}
for name, anno in laugh.items():
  if len(anno) > 0:
    valid_name[name] = 1
# list of all new segments
segments_new = {k: v for k, v in segments_new.items()
                if v in valid_name}
laugh = {k: v for k, v in laugh.items()
         if k in valid_name}
topic = {k: v for k, v in topic.items()
         if k in valid_name}
print(list(laugh.items())[8])
print(list(topic.items())[12])
print(topic['est/C_05_FF_06_05.10.wav'])
print("#Laugh", len(laugh))
print("#Topic", len(topic))

sami_factor = {
    'sam/08_VV': 2,
    'sam/04_IS': 2,
    'sam/03_IV': 2,
    'sam/07_SX': 2,
    'sam/02_V': 3,
    'sam/06_PS': 2,
    'sam/05_TP': 2,
    'sam/01_S': 3,
}
duration = {name: (d / sami_factor[name])
            if name in sami_factor else d
            for name, d in duration.items()}
# ===========================================================================
# Processing
# ===========================================================================
DEBUG = args.debug
bnf_network = N.models.BNF_2048_MFCC39()
delta_delta_feat = ('mfcc', 'mspec', 'pitch',
                    'f0', 'loudness', 'energy', 'sap')
# ====== FEAT ====== #
pipeline = pp.make_pipeline(steps=[
    pp.speech.AudioReader(remove_dc_n_dither=True, preemphasis=0.97),
    pp.base.NameConverter(converter=segments_new, keys='path'),
    pp.speech.SpectraExtractor(frame_length=FRAME_LENGTH,
                               step_length=STEP_LENGTH,
                               nfft=512, nmels=40, nceps=13,
                               fmin=FMIN, fmax=FMAX),
    pp.speech.SADextractor(nb_mixture=3, nb_train_it=25,
                           smooth_window=8, feat_name='energy'),
    # 'f0', 'pitch'
    pp.speech.openSMILEpitch(frame_length=0.03, step_length=STEP_LENGTH,
                             fmin=24, fmax=600, voicingCutoff_pitch=0.7,
                             f0min=64, f0max=420, voicingCutoff_f0=0.55,
                             f0=True, loudness=False, voiceProb=True,
                             method='acf'),
    pp.speech.openSMILEloudness(frame_length=FRAME_LENGTH,
                                step_length=STEP_LENGTH,
                                nmel=40, fmin=20, fmax=None,
                                to_intensity=False),
    # ====== bottleneck ====== #
    pp.base.DuplicateFeatures(name='mfcc', new_name='mfcc1'),
    pp.base.DeltaExtractor(width=9, order=(0, 1, 2), feat_name='mfcc1'),
    pp.base.StackFeatures(context=10, feat_name='mfcc1'),
    pp.speech.BNFExtractor(input_feat='mfcc1', network=bnf_network, pre_mvn=True),
    pp.base.RemoveFeatures(feat_name='mfcc1'),
    # ====== normalization ====== #
    pp.speech.RASTAfilter(rasta=True, sdc=0, feat_name='mfcc'),
    pp.base.DeltaExtractor(width=9, order=(0, 1, 2), feat_name=delta_delta_feat),
    pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=False,
                           feat_name=('mspec', 'spec', 'mfcc', 'bnf')),
    # ====== post processing ====== #
    pp.base.EqualizeShape0(feat_name=('spec', 'mspec', 'mfcc', 'energy',
                                      'pitch', 'f0', 'loudness', 'sap',
                                      'bnf', 'sad')),
    pp.base.AsType(type_map={'spec': 'float32', 'mspec': 'float32',
                             'mfcc': 'float32', 'energy': 'float32',
                             'pitch': 'float32', 'loudness': 'float32',
                             'sap': 'float32', 'bnf': 'float32',
                             'f0': 'float32', 'raw': 'float16'})
],
debug=DEBUG)
# ====== Debug mode ====== #
# 01_20140226b_S-2.1.wav
# bad: 01_20140226b_S-2.12.wav
if DEBUG:
  for f in sorted(list(segments_new.keys()))[11:12]:
    name = os.path.basename(f)
    # add feat
    feat = pipeline.transform(f)
    n = feat['spec'].shape[0]
    # delta_delta features
    for i in delta_delta_feat:
      X = feat[i]
      j = X.shape[1] // 3
      feat[i] = X[:, :j]
      feat[i + '_d1'] = X[:, j:2 * j]
      feat[i + '_d2'] = X[:, 2 * j:]
    # add laughter
    laughter = np.zeros(shape=(n,))
    for s, e, lt, spk in [v for k, v in laugh.items()
                          if name in k][0]:
      laughter[s:e] = 1
    feat['Laughter'] = laughter
    # save plot
    V.plot_features(feat, fig_width=8, title=f)
  V.plot_save('/tmp/tmp.pdf')
# ====== RUN mode ====== #
else:
  input(ctext("Do you want to delete old features?", 'red'))
  feat = pp.FeatureProcessor(jobs=list(segments_new.keys()),
                             extractor=pipeline, path=featpath,
                             ncache=180, ncpu=None, override=True)
  feat.run()
  pp.validate_features(feat, '/tmp/digisami_feat',
                       nb_samples=12, override=True)
  # ====== save ====== #
  with open(os.path.join(featpath, 'laugh'), 'wb') as f:
    cPickle.dump(laugh, f, protocol=2)
  with open(os.path.join(featpath, 'topic'), 'wb') as f:
    cPickle.dump(topic, f, protocol=2)
  with open(os.path.join(featpath, 'gender'), 'wb') as f:
    cPickle.dump(gender, f, protocol=2)
  with open(os.path.join(featpath, 'utt_duration'), 'wb') as f:
    cPickle.dump(duration, f, protocol=2)
  # ====== final check ====== #
  ds = F.Dataset(featpath, read_only=True)
  print(ds)
  ds.close()
  # ====== calculate pca ====== #
  pp.calculate_pca(featpath,
                   feat_name=('mspec', 'mfcc', 'spec', 'bnf',
                              'pitch', 'f0', 'energy', 'loudness'),
                   override=True)
