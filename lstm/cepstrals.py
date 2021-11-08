# This file extracts MFCCs from .wav files

import os
import librosa
import librosa.display
import pickle

cwd = os.getcwd()

prog_song_path = r'C:\uf-programming\cis4930\processed-datasets\wav\prog-chunks'
non_prog_song_path = r'C:\uf-programming\cis4930\processed-datasets\wav\non-prog-chunks'
OUT_FILE = r'C:\uf-programming\cis4930\processed-datasets\mfcc\prog'

mfcc_store = {'prog': {}, 'nonprog': {}}

prog_dir_size = len(os.listdir(prog_song_path))
non_prog_dir_size = len(os.listdir(non_prog_song_path))

count = 1
# List directories of test_prog songs
for path in os.listdir(prog_song_path):
    if path.startswith('.'): continue # Skip dotfiles

    print("Starting chunk dir #{}".format(count))

    song_dir = prog_song_path + '/' + path
    chunk_count = len(os.listdir(song_dir))

    mfcc_store['prog'][path] = []

    

    # chunk_count - 1 to discard MFCC-size
    for idx in range(0, chunk_count - 1):
        # Load chunk in Librosa
        chunk_name = song_dir + '/' + "__{}__".format(idx) + path + '.wav'

        y, sr = librosa.load(chunk_name, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_store['prog'][path].append(mfccs)
        
        print("On dir {}/{}".format(count, prog_dir_size))
        print("Prog: Done with {}".format(chunk_name))
        
    count += 1
        
count = 1
# List directories of test_prog songs
for path in os.listdir(non_prog_song_path):
    
    if path.startswith('.'): continue # Skip dotfiles

    print("Starting chunk dir #{}".format(count))

    song_dir = non_prog_song_path + '/' + path
    chunk_count = len(os.listdir(song_dir))

    mfcc_store['nonprog'][path] = []

    

    # chunk_count - 1 to discard MFCC-size
    for idx in range(0, chunk_count - 1):
        # Load chunk in Librosa
        chunk_name = song_dir + '/' + "__{}__".format(idx) + path + '.wav'

        y, sr = librosa.load(chunk_name, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_store['nonprog'][path].append(mfccs)

        print("On dir {}/{}".format(count, non_prog_dir_size))
        print("Non-Prog: Done with ", chunk_name)
    count += 1

print("Pickling...")
pickle.dump(mfcc_store, open(OUT_FILE, 'wb'))
