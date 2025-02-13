import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model and scaler
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']
    scaler = model_dict['scaler']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands with multi_hand=True
hands = mp_hands.Hands(static_image_mode=True,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3,
                       max_num_hands=2)

labels_dict = {0: 'A', 1: 'B', 2: 'C'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Draw all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect coordinates
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Process landmarks for the current hand
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Pad data if needed
        if len(results.multi_hand_landmarks) < 2:
            padding = 42 * 2 - len(data_aux)  # 42 landmarks per hand, 2 coordinates per landmark
            data_aux.extend([0.0] * padding)

        # Make prediction
        data_aux = np.asarray(data_aux).reshape(1, -1)
        data_aux_scaled = scaler.transform(data_aux)
        prediction = model.predict(data_aux_scaled)
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box around all detected hands
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()