import librosa
import utils
import featureEvaluation
import json
import math
import numpy as np
import sys
from tqdm import tqdm
import time
import os

FEATURE_NAMES = []
FEATURE_NAMES_INSERTED = False


def extract_spectral_contrast(fname, ext):
    features = []
    if ext == 'ods':
        return None

    if ext != 'mp3' and ext != 'm4a':
        data, sample_rate = librosa.load(fname, duration=30)
    else:
        data, sample_rate = utils.__audioread_load(
            fname, offset=0.0, duration=30, dtype=np.float32)
        data = utils.to_mono(data)

    # SPECTRAL CONTRASTS
    spectral_contrast = librosa.feature.spectral_contrast(data, sample_rate)
    mean_spectral_contrast = np.mean(spectral_contrast, axis=1)
    features.extend(mean_spectral_contrast)
    return features


def extract_features(fname, ext):
    global FEATURE_NAMES
    global FEATURE_NAMES_INSERTED

    if ext == 'ods':
        return None

    features = []

    # DURATION
    with utils.audio_open(fname) as input_file:
        duration = input_file.duration
        features.append(duration)
        if not FEATURE_NAMES_INSERTED:
            FEATURE_NAMES.append('Duration')

    if ext != 'mp3' and ext != 'm4a':
        data, sample_rate = librosa.load(fname, duration=30)
    else:
        data, sample_rate = utils.__audioread_load(
            fname, offset=0.0, duration=30, dtype=np.float32)
        data = utils.to_mono(data)

    # MEAN AMPLITUDE
    mean_amplitude = np.mean(np.array(data))
    features.append(mean_amplitude)
    if not FEATURE_NAMES_INSERTED:
        FEATURE_NAMES.append('Mean Amplitude')

    # ROOT MEAN SQUARE ENERGY
    rmse = librosa.feature.rms(data)
    mean_rmse = np.mean(rmse)
    features.append(mean_rmse)
    if not FEATURE_NAMES_INSERTED:
        FEATURE_NAMES.append('Mean RMSE')

    # ZERO CROSSING RATE: Number of times the signal changes sign
    zero_crossings = librosa.zero_crossings(data, pad=False)
    zcr = sum(zero_crossings)
    features.append(zcr)
    if not FEATURE_NAMES_INSERTED:
        FEATURE_NAMES.append('ZCR')

    # SPECTRAL CENTROID: Weighted mean of frequencies
    spectral_centroids = librosa.feature.spectral_centroid(
        data, sr=sample_rate)
    mean_spectral_centroid = np.mean(spectral_centroids)
    features.append(mean_spectral_centroid)
    if not FEATURE_NAMES_INSERTED:
        FEATURE_NAMES.append('Mean Spectral Centroid')

    # SPECTRAL ROLLOFF: Frequency below which x% of spectral energy lies
    rolloff_percentages = [element / 10.0 for element in range(1, 10, 1)]
    for rp in rolloff_percentages:
        spectral_rolloff = librosa.feature.spectral_rolloff(
            data, sr=sample_rate, roll_percent=rp)
        mean_spectral_rolloff = np.mean(spectral_rolloff)
        features.append(mean_spectral_rolloff)
        if not FEATURE_NAMES_INSERTED:
            FEATURE_NAMES.append(f'Mean Spectral Rolloff RP={rp}')

    # MFCC: Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(data, sr=sample_rate)
    mean_mfccs = np.mean(mfccs, axis=1)
    features.extend(mean_mfccs)
    if len(mean_mfccs) != 20:
        length = len(mean_mfccs)
        print(f'ERROR: Length of mean_mfccs for {fname} is {length}!')

    if not FEATURE_NAMES_INSERTED:
        for i in range(len(mean_mfccs)):
            FEATURE_NAMES.append(f'Mean MFCC {i+1}')

    # SPECTRAL CONTRASTS
    spectral_contrast = librosa.feature.spectral_contrast(data, sample_rate)
    mean_spectral_contrast = np.mean(spectral_contrast, axis=1)
    features.extend(mean_spectral_contrast)
    if not FEATURE_NAMES_INSERTED:
        for i in range(len(mean_spectral_contrast)):
            FEATURE_NAMES.append(f'Mean Spectral Contrast {i+1}')

    FEATURE_NAMES_INSERTED = True
    print(FEATURE_NAMES)
    return features


def add_data(path, label, X, y):
    for i, fname in enumerate(tqdm(os.listdir(path)[:])):
        if fname.split('.')[-1] == 'ods':
            continue
        # t1 = time.time()
        try:
            features = extract_features(
                path + '/' + fname, fname.split('.')[-1])
        except Exception as e:
            print(e, fname)
            continue
        features = np.array(features)
        if features.all():
            X.append(features)
            y.append(label)
            # t2 = time.time()
            # print(t2 - t1)
            # sys.exit()
        # if (i + 1) % 10 == 0:
        #     folder = path.split('/')[-1]
        #     print(f'Inserted {(i + 1)} songs from {folder}.')

    return X, y


def make_features(save_features=False, X_name='X', y_name='y'):
    X = []
    y = []

    path = './CIS4930fa21_training_set/Progressive_Rock_Songs'
    X, y = add_data(path, 1, X, y)

    path = './CIS4930fa21_training_set/Not_Progressive_Rock/Other_Songs'
    X, y = add_data(path, 0, X, y)

    path = './CIS4930fa21_training_set/Not_Progressive_Rock/Top_Of_The_Pops'
    X, y = add_data(path, 0, X, y)

    if save_features:
        with open(f'{X_name}.txt', 'w') as X_file:
            X_file.write(str(X))
        with open(f'{y_name}.txt', 'w') as y_file:
            y_file.write(str(y))
        with open('feature_names.txt', 'w') as feature_name_file:
            feature_name_file.write(str(FEATURE_NAMES))

        with open(f'{X_name}.json', 'w') as X_file:
            json.dump(np.array(X).tolist(), X_file)
        with open(f'{y_name}.json', 'w') as y_file:
            json.dump(y, y_file)
        with open('feature_names.json', 'w') as feature_name_file:
            json.dump(FEATURE_NAMES, feature_name_file)

    return X, y


if __name__ == '__main__':
    t1 = time.time()
    X, y = make_features(save_features=True)
    t2 = time.time()
    print()
    print(f'Collecting features took {t2 - t1} seconds')
    print()
    featureEvaluation.assess_features(X, y, FEATURE_NAMES)
