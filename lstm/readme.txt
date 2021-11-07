These files are scripts that represent the pipeline for the LSTM Embeddings portion of the Clockwork Angels report.

They should be filled in with directories specific to your machine. They should also be run in the order:

1. convert.py - Converts .mp3, .flac, etc to .wav
2. chunker.py - Converts these .wav files into 3-second long .wav file 'chunks'
3. cepstrals.py - Extracts the Mel-frequency cepstral coefficients and saves them to a pickle file
4. memory.py - Trains an LSTM on each of the songs (their MFCCs)
5. classifier.py - The CNN classifier to be trained on the LSTM weights
