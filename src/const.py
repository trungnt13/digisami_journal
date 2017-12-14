from __future__ import print_function, division, absolute_import
import os

basedir = '/media/trung/data1'
if not os.path.exists(basedir):
  basedir = '/mnt/sdb1'

inpath = [
    os.path.join(basedir, "digisami_data/estonian"),
    os.path.join(basedir, "digisami_data/finnish"),
    os.path.join(basedir, "digisami_data/sami_conv")
]
segpath = '/home/trung/data/sami_segs'
featpath = "/home/trung/data/sami_feat"

SR = 16000
CUT_DURATION = 30
FRAME_LENGTH = 0.025
STEP_LENGTH = 0.01

def utt_id(name):
  """ Return unique utterance ID for given segment"""
  lang, name = name.split('/')
  name = name.split('.')[0]
  if lang == 'sam':
    _ = name.split('_')
    name = _[0] + '_' + _[-1].split('-')[0]
  if lang == 'fin':
    name = name[:3]
  if lang == 'est':
    name = name[:4]
  return lang + '/' + name
