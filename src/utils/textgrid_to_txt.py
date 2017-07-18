from __future__ import print_function, division, absolute_import

import os
import shutil

from odin.utils import get_all_files
from odin.preprocessing import textgrid

from collections import defaultdict

dataset = [
    "sami_conv",
    "estonian",
    "finnish",
]
datatype = [
    "trans",
    "laugh",
    "topic"
]

path = "/Volumes/backup/digisami_data"
ignore_files = []
ignore_ds = []


# ===========================================================================
# Helper function
# ===========================================================================
def get_tier_information(fname, ds, dt, tier_name):
    """ Infer tier information: speaker ID, language
    from tier_name

    Return
    ------
    speaker_ID, language (eng, est, sam, fin)
    """
    speakerID = None
    langCODE = None
    #Estonia
    if ds == dataset[1]:
        _ = fname.split('_')
        gender, s1, s2 = _[2], _[3], _[4]
        s1 += gender[0]
        s2 += gender[1]
        if dt == datatype[0] and tier_name in ('S1Est', 'S1Eng', 'S2Est', 'S2Eng'):
            speakerID = s1 if 'S1' in tier_name else s2
            langCODE = 'eng' if 'Eng' in tier_name else 'est'
        elif dt == datatype[1] and tier_name in ('S1Laugh', 'S2Laugh'):
            speakerID = s1 if 'S1' in tier_name else s2
            langCODE = ''
        elif dt == datatype[2] and tier_name in ('topic'):
            speakerID = ''
            langCODE = ''
    # Finnish
    elif ds == dataset[2]:
        _ = fname.split('_')
        gender, s1, s2 = _[3], _[4], _[5]
        s1 += gender[0]
        s2 += gender[1]
        if dt == datatype[0] and tier_name in ('puhujaA', 'speakerA', 'puhujaB',
                                               'speakerB'):
            speakerID = s1 if 'A' == tier_name[-1] else s2
            langCODE = 'fin' if 'puhuja' in tier_name else 'eng'
        elif dt == datatype[1] and tier_name in ('laughterA', 'laughterB'):
            speakerID = s1 if 'A' == tier_name[-1] else s2
            langCODE = ''
        elif dt == datatype[2] and tier_name in ('topic'):
            speakerID = ''
            langCODE = ''
    # Sami
    elif ds == dataset[0]:
        if dt == datatype[0] and \
        tier_name in ('lauseet', 'suomennos'):
            speakerID = fname.split('_')[-1]
            langCODE = 'fin' if 'suomennos' in tier_name else 'sam'
        elif dt == datatype[1] and tier_name in ('laugh'):
            speakerID = fname.split('_')[-1]
            langCODE = ''
        elif dt == datatype[2] and tier_name in ('topic'):
            speakerID = ''
            langCODE = ''
    # return the results
    if speakerID is None or langCODE is None:
        return None
    return speakerID, langCODE


def filter_trans(t):
    # ====== misspell ====== #
    misspell = {
        'umm': 'um',
        'um': 'um',
        'mhm': 'um',
        'y-you': 'you',
        'yeahrs': 'years',
        'yeahr': 'year',
        'whare': 'where',
        "voluntee": "volunteer",
        'usally': 'usually',
        "Tueasday": "Tuesday",
        "Sudnday": "Sunday",
        "stuying": "studying",
        "somewehre": "somewhere",
        "sgain": "again",

        "Unfortunenately": "Unfortunately",
        "Fortunenately": "Fortunately",

        "scolarship": "scholarship",
        "scholarshop": "scholarship",

        "prese": "press",
        "polotics": "politics",
        "oportunity": "opportunity",
        "morphsynthesis": "morphology synthesis",
        "morphanalysis": "morphology analysis",
        "managaged": "managed",
        "kindegartens": "kindergartens",
        "intitute": "institute",
        "inmathematics": "in mathematics",
        "hostpitals": "hospitals",
        "horisontal": "horizontal",
        "gymnasi": "gymnastic",
        "gocery": "grocery",
        "frameanalysis": "frame analysis",
        "fortu": "fortune",
        "excercised": "exercised",

        "aaand": "and",
        "pressnt": "present",
        "webmedia": "web media",

        "exatcly": "exactly",
        "Exacty": "exactly",
        "exactl": "exact",

        "okey": "ok",
        "oke": "ok",
        "okei": "ok",

        "dialoogid": "dialog ID",
        "dather": "father",
        "culatively": "cumulatively",
        "competetions": "competitions",
        "cinamon": "cinnamon",
        "buut": "but",
        "bussinessmen": "business men",
        "binded": "bind",
        "bevause": "because",
        "bacherlor": "bachelor",
        "bacause": "because",
        "algthough": "although",

        "questionns": "questions",
        "questio": "question",

        "interesing": "interesting",
        "certanly": "certainly",
        "biotechincal": "bio-technical",
        "youh": "youth",
        "valueable": "valuable",

        "yees": "yes",
        "yeeaa": "yeah",
        'yea': 'yeah',
        'yeahh': 'yeah',
        "yeahah": "yeah",
        "yeyeah": "yeah",
        "yesyes": "yes",
        "yeahyeah": "yeah",
        "yeahm": "yeah",
        "yeahaah": "yeah",
        "yeahaaah": "yeah",
        "yeaha": "yeah",

        "acutally": "actually",
        "actualy": "actually",
        "actally": "actually",
        "acually": "actually",

        "alreayd": "already",
        "basicly": "basically",
        "bussiness": "business",

        "stu-": "student",
        "lang-": "language",
        "pressnt": "present",
        "ooh": "oh",
    }
    t = ' '.join([misspell[i] if i in misspell else i
                  for i in t.split(' ')])
    # ====== shortcut ====== #
    shortcut_words = {
        "can't": "cannot",
        "'s": " is",
        "'m": " am",
        "'d": " would",
        "'ve": " have",
        "n't": " not",
        "'re": " are",
        "'ll": " will",
        "nt't": " not"
    }
    for i, j in shortcut_words.iteritems():
        t = t.replace(i, j)
    # ====== other anno ====== #
    return t


def preprocessed_text(text, langCODE):
    pattern = u'"#$%&+/:;<=>@\\^_`{|}~\t\n'
    unicode_trans = dict((ord(char), u' ') for char in pattern)
    text = text.translate(unicode_trans)
    if langCODE == 'eng':
        text = filter_trans(text)
    return text.strip().encode('utf-8').decode('utf-8')


# ===========================================================================
# Processing
# ===========================================================================
# FOrmat of csv file:
# SpeakerID;start_time;end_time;text
for ds in dataset:
    if ds in ignore_ds: continue
    for dt in datatype:
        print("=======================================")
        print(" Processing dataset:", ds, "datatype:", dt)
        print("=======================================")
        inpath = os.path.join(path, ds, "%s_textgrid" % dt)
        outpath = os.path.join(path, ds, "%s_csv" % dt)
        # remove old processed data at outpath
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        os.mkdir(outpath)
        files = get_all_files(inpath, filter_func=lambda x: '.TextGrid' == x[-9:])
        # read each textgrid file
        for fpath in files:
            # Lang -> (speaker, start, end, text)
            data = defaultdict(list)
            fname = os.path.basename(fpath).replace('.TextGrid', '.csv')
            # read and process the textgrid file
            with open(fpath, 'r') as f:
                tg = textgrid.TextGrid(f)
                xmin = tg.xmin
                xmax = tg.xmax
                print("* File:", os.path.basename(fpath),
                      "xmin:", xmin, "xmax:", xmax)
                # iterate over each tier
                for i in tg:
                    tier_name = i.tier_name()
                    info = get_tier_information(fname.replace('.csv', ''),
                        ds, dt, tier_name)
                    # # no information returned
                    if info is None: continue
                    else: speakerID, langCODE = info
                    # print(tier_name, speakerID, langCODE)
                    for start, end, text in i:
                        # ignore NULL text
                        if len(text) == 0: continue
                        d = [float(start), float(end), preprocessed_text(text, langCODE)]
                        if len(speakerID) > 0:
                            d = [speakerID] + d
                        data[langCODE].append(d)
            # Sort by start time
            data = {lang: sorted(dat, key=lambda x: x[len(x) - 3])
                    for lang, dat in data.iteritems()}
            if len(data) == 0: # skip if no data points found
                continue
            elif len(data) == 1:
                data = data.values()[0]
            else:
                data = [([lang] if len(lang) > 0 else []) + i
                    for lang, dat in data.iteritems() for i in dat]
            title = ['LangCODE', 'SpeakerID', 'StartTime', 'EndTime', 'Text']
            # save to CSV file
            with open(os.path.join(outpath, fname), 'w') as f:
                f.write("xmin:%f\n" % xmin)
                f.write("xmax:%f\n" % xmax)
                header = ':'.join(title[(len(title) - len(data[0])):]) + '\n'
                f.write(header)
                for dat in data:
                    dat = [str(i) for i in dat[:-1]] + [dat[-1].encode('utf8')]
                    f.write(':'.join(dat) + "\n")

# ===========================================================================
# Validate filenames
# ===========================================================================
for ds in dataset:
    if ds == dataset[0]: continue
    if ds in ignore_ds: continue
    for dt in datatype:
        print("Validating data:", ds, dt)
        audiopath = os.path.join(path, ds, "audio")
        audiofiles = get_all_files(audiopath, filter_func=lambda x: '.wav' == x[-4:])
        audiofiles = [os.path.basename(i).replace('.wav', '') for i in audiofiles]
        videopath = os.path.join(path, ds, "video")
        videofiles = get_all_files(videopath, filter_func=lambda x: '.mp4' == x[-4:])
        videofiles = [os.path.basename(i).replace('.mp4', '')
                      for i in videofiles]

        outpath = os.path.join(path, ds, "%s_csv" % dt)
        annofiles = get_all_files(outpath, filter_func=lambda x: '.csv' == x[-4:])
        annofiles = [os.path.basename(i).replace('.csv', '')
                     for i in annofiles]

        if ds != dataset[0]:
            assert len(audiofiles) == len(videofiles), \
                '%d != %d' % (len(audiofiles), len(videofiles))
            assert set(annofiles) == set(videofiles)
        assert len(audiofiles) == len(annofiles), \
            '%d != %d' % (len(audiofiles), len(annofiles))
        assert set(annofiles) == set(audiofiles)
