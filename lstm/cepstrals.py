# This file extracts MFCCs from .wav files

import os
import librosa
import librosa.display
import pickle

cwd = os.getcwd()

song_path = ''
OUT_FILE = ''

mfcc_store = {'prog': {}, 'nonprog': {}}

count = 1
# List directories of test_prog songs
for path in os.listdir(song_path):
    if path.startswith('.'): continue # Skip dotfiles

    print("Starting chunk dir #{}".format(count))

    song_dir = song_path + '/' + path
    chunk_count = len(os.listdir(song_dir))

    mfcc_store['nonprog'][path] = []

    count += 1

    # chunk_count - 1 to discard MFCC-size
    for idx in range(0, chunk_count - 1):
        # Load chunk in Librosa
        chunk_name = song_dir + '/' + "__{}__".format(idx) + path + '.wav'

        y, sr = librosa.load(chunk_name, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_store['nonprog'][path].append(mfccs)

        print("Done with ", chunk_name)

print("Pickling...")
pickle.dump(mfcc_store, open(OUT_FILE, 'wb'))
