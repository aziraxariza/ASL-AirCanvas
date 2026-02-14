import cv2
import mediapipe as mp
import csv
import os
import string

# ====== CONFIGURATION ======
letters = list(string.ascii_uppercase)  # A-Z
words = ["HELLO", "YES", "NO", "THANKYOU"]
labels = letters + words

samples_per_label = 300  # change if needed
# ============================

current_label_index = 0
current_label = labels[current_label_index]
count = 0

print("Labels to collect:", labels)
print("Press 'n' to move to next label")
print("Press ESC to exit")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

file_exists = os.path.isfile("dataset.csv")

with open("dataset.csv", mode="a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and count < samples_per_label:
            for handLms in result.multi_hand_landmarks:
                landmark_list = []

                for lm in handLms.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                landmark_list.append(current_label)
                writer.writerow(landmark_list)

                count += 1
                print(f"{current_label}: {count}/{samples_per_label}")

        # Display info on screen
        cv2.putText(frame, f"Label: {current_label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame, f"Samples: {count}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.imshow("ASL Data Collection", frame)

        key = cv2.waitKey(1)

        # Move to next label
        if key == ord('n'):
            current_label_index += 1
            if current_label_index >= len(labels):
                print("All labels completed.")
                break

            current_label = labels[current_label_index]
            count = 0
            print(f"\nNow collecting for: {current_label}")

        # Exit
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
