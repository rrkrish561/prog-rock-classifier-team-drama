# This file is used for converting songs into .wav files

import os
from pydub import AudioSegment

TEST_PROG_IN_PATH = r'C:\uf-programming\cis4930\datasets\Progressive_Rock_Songs'
TEST_PROG_OUT_PATH = r'C:\uf-programming\cis4930\processed-datasets\wav\prog-rock'

TEST_NON_PROG_IN_PATH = r'C:\uf-programming\cis4930\datasets\Other_Songs'
TEST_NON_PROG_OUT_PATH = r'C:\uf-programming\cis4930\processed-datasets\wav\non-prog-rock'

TEST_OTHER_IN_PATH = ''
TEST_OTHER_OUT_PATH = ''

#Converting songs to .wav
for file in os.scandir(TEST_NON_PROG_IN_PATH):
    base_name = os.path.splitext(os.path.basename(file.path))[0]
    
    base_name_alnum = ""
    for character in base_name:
        if character.isalnum():
            base_name_alnum += character
    
    out_file = os.path.join(TEST_NON_PROG_OUT_PATH, base_name_alnum + ".wav")
    try:
        AudioSegment.from_file(file.path).export(out_file, format="wav")
    except:
        print("skipping" + file.path)
        continue
    print(f"Creating {out_file}")
