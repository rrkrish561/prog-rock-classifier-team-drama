import os
import cv2 as cv
import numpy as np
import json


def collect_image_data(height, width):
    X, y = [], []

    path = 'data/spectral_contrast_images/0'

    for fname in os.listdir(path):
        image = cv.imread(path + '/' + fname)
        if image is None:
            print('Error:', fname)
        image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
        X.append(image)
        y.append(0.0)

    path = 'data/spectral_contrast_images/1'

    for fname in os.listdir(path):
        image = cv.imread(path + '/' + fname)
        image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
        X.append(image)
        y.append(1.0)

    X, y = np.asarray(X), np.asarray(y)

    X = np.moveaxis(X, -1, 1)

    # with open('./spectral contrast cnn/spectral_contrast_data/imagedata.json', 'w') as datafile:
    #     json.dump(X.tolist(), datafile)

    # with open('./spectral contrast cnn/spectral_contrast_data/imagelabels.json', 'w') as yfile:
    #     json.dump(y.tolist(), yfile)

    return X, y


if __name__ == '__main__':
    collect_image_data(500, 500)
