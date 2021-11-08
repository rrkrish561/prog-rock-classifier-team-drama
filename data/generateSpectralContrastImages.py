import librosa
import utils
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from PIL import Image
import os
from tqdm import tqdm


def generate_spectral_contrast(fname, ext, label):
    if ext == 'ods':
        return None

    if ext != 'mp3' and ext != 'm4a':
        data, sample_rate = librosa.load(fname, duration=30)
    else:
        data, sample_rate = utils.__audioread_load(
            fname, offset=0.0, duration=30, dtype=np.float32)
        data = utils.to_mono(data)
    try:
        spectral_contrast = librosa.feature.spectral_contrast(
            data, sample_rate)
    except:
        print(fname)
        return
    normalized_spectral_contrast = minmax_scale(
        spectral_contrast, feature_range=(0, 1), axis=1)
    # print(normalized_spectral_contrast)
    # img = Image.fromarray(np.uint8(cm.viridis(normalized_spectral_contrast) * 255))
    img_name = ''.join(fname.split('.')[:-1]).split('/')[-1]
    plt.imsave(f'./spectral_contrasts/{label}/{img_name}.jpg',
               normalized_spectral_contrast, format='jpg', cmap='viridis')


def create_dataset(path, label):
    for fname in tqdm(os.listdir(path)[:]):
        generate_spectral_contrast(
            path + '/' + fname, fname.split('.')[-1], label)


if __name__ == '__main__':
    path = './CIS4930fa21_training_set/Progressive_Rock_Songs'
    create_dataset(path, 1)

    path = './CIS4930fa21_training_set/Not_Progressive_Rock/Other_Songs'
    create_dataset(path, 0)

    path = './CIS4930fa21_training_set/Not_Progressive_Rock/Top_Of_The_Pops'
    create_dataset(path, 0)
