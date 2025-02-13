import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 500  # Increased to 500 images per class

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))
    print(f'Press "Q" when ready to start collecting {dataset_size} images')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f'Ready? Press "Q" to collect {dataset_size} images for class {j}!',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f'Collecting: {counter}/{dataset_size}',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Save image
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        counter += 1

        # Add small delay to avoid duplicate images
        time.sleep(0.1)

        if cv2.waitKey(1) == ord('q'):  # Allow early stopping
            break

    print(f'Finished collecting {counter} images for class {j}')

cap.release()
cv2.destroyAllWindows()