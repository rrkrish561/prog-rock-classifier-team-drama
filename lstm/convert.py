# This file is used for converting songs into .wav files

import os
from pydub import AudioSegment

TEST_PROG_IN_PATH = ''
TEST_PROG_OUT_PATH = ''

TEST_NON_PROG_IN_PATH = ''
TEST_NON_PROG_OUT_PATH = ''

TEST_OTHER_IN_PATH = ''
TEST_OTHER_OUT_PATH = ''

#Converting songs to .wav
for file in os.scandir(AUG_NONPROG_IN_PATH):
    out_file = os.path.join(AUG_NONPROG_OUT_PATH, os.path.splitext(os.path.basename(file.path))[0] + ".wav")
    try:
        AudioSegment.from_file(file.path).export(out_file, format="wav")
    except:
        print("skipping" + file.path)
        continue
    print(f"Creating {out_file}")
