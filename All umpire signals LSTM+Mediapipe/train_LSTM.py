import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import tensorflow as tf
import threading
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# List of umpire signals
signals = [
    "Out", "No-ball", "Free-Hit", "Wide ball", "Four runs", "Six Runs", "Byes", "Leg Byes", "Bouncer",
    "Television /3rd Umpire", "Dead Ball", "Short Run", "Penalty Runs", "Revoke Decision", "Powerplay",
    "Soft Signal"
]

X, y = [], []
no_of_timesteps = 10

for idx, signal in enumerate(signals):
    file_path = rf"C:\PROJECT\Umpire Environment\Data\{signal}.txt"
    df = pd.read_csv(file_path)
    dataset = df.iloc[:, 1:].values  # Exclude index column
    n_sample = len(dataset)
    
    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i, :])
        y.append(idx)  # Assign class index

X, y = np.array(X), np.array(y)

# Convert labels to one-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y.reshape(-1, 1))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(len(signals), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
model.save("Cricket_LSTM_Model.h5")