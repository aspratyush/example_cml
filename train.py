from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

################## Test 1 ####################
seed = 42
# Generate data
X, y = make_classification(n_samples = 1000, random_state=seed)
# Make a train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=seed)

# Fit a model
depth = 2
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

acc = clf.score(X_test, y_test)
print(acc)
with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")

# Plot it
disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')


################## Test 2 ####################
params = {
        'filter': [64, 128],
        'kernel': 2,
        'dense': [50, 1],
        'batch_size': 16,
        'epochs': 1,
        'steps': 50,
        'features': 4,
        'patience': 2,
        'min_delta': 0.0001
    }
n_steps = params['steps']
n_features = params['features']
# Model
model = tf.keras.Sequential()
model.add(layers.Input(shape=(n_steps, n_features, 1)))
model.add(layers.Conv2D(filters=params['filter'][0], kernel_size=params['kernel'], activation='relu'))
model.add(layers.Conv2D(filters=params['filter'][1], kernel_size=params['kernel'], activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(params['dense'][0], activation='relu'))
model.add(layers.Dense(params['dense'][1], activation='sigmoid'))
model.compile(optimizer='adam', loss='mse')
model.save_weights('model.h5')
