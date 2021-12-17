import json
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

tf.get_logger().setLevel('ERROR')

X = pd.read_json('./data/features-and-labels/X.json')
Y = pd.read_json('./data/features-and-labels/y.json')
X_test = pd.read_json('./data/features-and-labels/X_test.json')
Y_test = pd.read_json('./data/features-and-labels/y_test.json')

# X = X.iloc[:, 32:]
# features selected based on importance > 0.02
X = X[[0, 2, 5, 14, 16, 19, 20, 23, 28, 34, 39, 40]]
# > 0.025
X = X[[0, 2, 14, 23, 28, 34, 40]]
X_test = X_test[[0, 2, 14, 23, 28, 34, 40]]

tf.random.set_seed(0)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.3, random_state=42)

Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
Y_test = to_categorical(Y_test)

X_train = X_train.values
X_val = X_val.values
X_test = X_test.values

# DNN
modelDNN = keras.Sequential([
    # 1st hidden layer
    keras.layers.Dense(512, activation="relu",
                       input_dim=X_train.shape[1], kernel_regularizer=keras.regularizers.l2(0.001)),
    # keras.layers.Dropout(0.5),

    # 2nd hidden layer
    keras.layers.Dense(256, activation="relu",
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    # keras.layers.Dropout(0.5),

    # 3rd hidden layer
    keras.layers.Dense(64, activation="relu",
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    # keras.layers.Dropout(0.5),

    # output layer
    keras.layers.Dense(2, activation="sigmoid")])

#optimizer = keras.optimizers.Adam(learning_rate=0.1)
modelDNN.compile(optimizer="adam",
                 loss="binary_crossentropy",
                 metrics=["accuracy"])

history = modelDNN.fit(X_train, Y_train,
                       epochs=150,
                       batch_size=15,
                       shuffle=True,
                       class_weight={0: 1, 1: 1.5},
                       verbose=0)

modelDNN.evaluate(X_train, Y_train)
modelDNN.evaluate(X_val, Y_val)
modelDNN.evaluate(X_test, Y_test)

pred_train = modelDNN.predict_classes(X_train)
pred_val = modelDNN.predict_classes(X_val)

cm0 = confusion_matrix(np.argmax(Y_train, axis=1), pred_train)
fig0 = plt.figure()
sns.heatmap(cm0, annot=True)
plt.show()

cm1 = confusion_matrix(np.argmax(Y_val, axis=1), pred_val)
fig1 = plt.figure()
sns.heatmap(cm1, annot=True)
plt.show()

explainer = shap.DeepExplainer(modelDNN, X_train)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, plot_type="bar", max_display=32)
