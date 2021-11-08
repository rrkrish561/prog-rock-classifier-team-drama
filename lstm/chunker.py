# This file splits each .wav file into 3 second 'chunks'

import os
from pydub import AudioSegment
from pydub.utils import make_chunks

cwd = os.getcwd()
PROG_WAV_PATH = r'C:\uf-programming\cis4930\processed-datasets\wav\prog-rock'
NON_PROG_WAV_PATH = r'C:\uf-programming\cis4930\processed-datasets\wav\non-prog-rock'

PROG_CHUNK_PATH = r'C:\uf-programming\cis4930\processed-datasets\wav\prog-chunks'
NON_PROG_CHUNK_PATH = r'C:\uf-programming\cis4930\processed-datasets\wav\non-prog-chunks'

TEST_PROG_WAV_PATH = cwd + ''
TEST_NON_PROG_WAV_PATH = cwd + ''
TEST_OTHER_WAV_PATH = cwd + ''

TEST_PROG_CHUNK_PATH = cwd + ''
TEST_NON_PROG_CHUNK_PATH = cwd + ''
TEST_OTHER_CHUNK_PATH = cwd + ''

CHUNK_LENGTH = 3000 # 3 second chunks

# chunkify prog songs
for file in os.scandir(NON_PROG_WAV_PATH):
    sound = AudioSegment.from_file(file.path, format='wav')
    chunks = make_chunks(sound, CHUNK_LENGTH)

    song_name = os.path.splitext(os.path.basename(file.path))[0]
    dir_name = NON_PROG_CHUNK_PATH + '/' + song_name

    # make directory for this song's chunks - name of song
    os.mkdir(dir_name)

    print("Chunking", song_name)

    # add chunks to that directory
    for idx, chunk in enumerate(chunks):
        chunk_name = dir_name + '/' + "__{}__".format(idx) + song_name + '.wav'
        chunk.export(chunk_name, format='wav')
