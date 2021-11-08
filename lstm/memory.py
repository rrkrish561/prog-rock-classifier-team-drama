import pickle
import numpy as np
import pdb
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

INFILE = r'C:\uf-programming\cis4930\processed-datasets\mfcc\mfcc_pickle'
OUTFILE = ""

# 'nonprog' or 'prog', depending
KEY = 'nonprog'

# Function to  split dataset into a supervised learning setup
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps

        if end_ix > len(sequences) - 1:
            break

        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Accumulates all MFCCs for the song from all of its 3-second 'chunks'
def create_dataset(collection, song_name):
    mfcc_chunks = collection[song_name]

    # transpose so each column is an mfcc
    # each row is a timestep
    accum = mfcc_chunks[0].T

    # acccumulate all chunks into a single chunk
    for chunk in mfcc_chunks[1:]:
        accum = np.vstack((accum, chunk.T))

    return accum

# This holds the W, U, tensors from the LSTM for each song
memory_datastore = {
        # song_name: { 'W': tensor, 'U': tensor }
}

mfcc = pickle.load( open(INFILE, 'rb') )



# UNITS = 200 # LSTM hidden unit count 

# # Get number of songs for progress indication
# song_count = len(mfcc[KEY])

# # Train an LSTM on each song, store the weights in pickle file
# for idx, song_name in enumerate(mfcc[KEY].keys()):
#     print("Processing {}/{}".format(idx + 1, song_count))

#     dataset = create_dataset(mfcc[KEY], song_name)

#     n_steps = 3
#     X, y = split_sequences(dataset, n_steps)
#     n_features = X.shape[2] # 20 MFCC coefficients

#     # Define model
#     model = Sequential()
#     model.add(
#         LSTM(
#             UNITS,
#             activation='relu',
#             input_shape=(n_steps, n_features)
#         )
#     )
#     model.add(Dense(n_features))
#     model.compile(optimizer='adam', loss='mse')

#     # Fit model. # 5 epochs seem to be convergence spot
#     history = model.fit(X, y, epochs=5)

#     W = model.layers[0].get_weights()[0]
#     U = model.layers[0].get_weights()[1]

#     memory_datastore[song_name] = { 'W': W, 'U': U }


# pickle.dump(memory_datastore, open(OUTFILE, 'wb'))
