
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Parameters
frame_height = 64
frame_width = 64
frame_count = 30
video_dir_normal = "Dataset/Normal"
video_dir_shoplifting = "Dataset/Shoplifting"

# Function to extract frames from video
def preprocess_video(video_path):
    frames = []
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        frame = frame / 255.0  # normalize
        frames.append(frame)
        if len(frames) == frame_count:
            break

    video.release()

    while len(frames) < frame_count:
        frames.append(np.zeros((frame_height, frame_width, 3)))

    return np.array(frames)

# Load the dataset
def load_dataset():
    X, y = [], []

    for file in os.listdir(video_dir_normal):
        if file.endswith(".mp4"):
            path = os.path.join(video_dir_normal, file)
            X.append(preprocess_video(path))
            y.append(0)

    for file in os.listdir(video_dir_shoplifting):
        if file.endswith(".mp4"):
            path = os.path.join(video_dir_shoplifting, file)
            X.append(preprocess_video(path))
            y.append(1)

    X = np.array(X)
    y = to_categorical(np.array(y), 2)
    return X, y

print("Loading dataset...")
X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(frame_count, frame_height, frame_width, 3)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=25, batch_size=8, validation_data=(X_test, y_test))

# Save the model
model.save("model.h5")
print("✅ Model saved as model.h5")
