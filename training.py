#importing dependencies
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

#loading data (numpy array)
array_X = np.load(r"C:\Users\MASTER\Sign_Language_Classification\X.npy")
array_y = np.load(r"C:\Users\MASTER\Sign_Language_Classification\Y.npy")

#rearranging array_X to match with array_y
X = np.concatenate((array_X[204:409,:],
            array_X[822:1028,:],
            array_X[1649:1855,:],
            array_X[1443:1649,:],
            array_X[1236:1443,:],
            array_X[1855:2062,:],
            array_X[615:822,:],
            array_X[409:615,:],
            array_X[1028:1236,:],
            array_X[0:204,:]),axis = 0)

#extracting class labels from array_y(sparse matrix)
y = np.argmax(array_y, axis = 1)

#splitting X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 42)

#building the neural network 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, Flatten
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu', input_shape = (64,64,1)))
model.add(AveragePooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu'))
model.add(AveragePooling2D())
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

#compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#fitting the model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

#saving model
model.save("digit_sign_language_class.h5", history)