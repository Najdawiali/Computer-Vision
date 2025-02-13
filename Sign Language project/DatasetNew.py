import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands with multi_hand=True
hands = mp_hands.Hands(static_image_mode=True,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3,
                       max_num_hands=2)  # Allow detection of up to 2 hands

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    print(f"Processing class {dir_}")
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Process all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                # First pass: collect all coordinates
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Second pass: normalize coordinates
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            # If we have less than 2 hands, pad with zeros
            if len(results.multi_hand_landmarks) < 2:
                padding = 42 * 2 - len(data_aux)  # 42 landmarks per hand, 2 coordinates per landmark
                data_aux.extend([0.0] * padding)

            data.append(data_aux)
            labels.append(dir_)

print(f"Processed {len(data)} images")

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)