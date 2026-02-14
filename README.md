# 🖐 ASL Recognition System with Air Canvas

A real-time American Sign Language (ASL) recognition system built using Computer Vision and Deep Learning.

This project detects hand gestures through a webcam, predicts ASL letters and selected words, converts them to text, and provides speech output. It also includes an interactive Air Canvas feature for gesture-based drawing.

---

## Features

- Real-time ASL letter recognition (A–Z)
- Detection of common words (HELLO, YES, NO, THANKYOU)
- 96%+ classification accuracy
- Confidence-based prediction filtering
- Word buffering system
- Text-to-speech output
- Air Canvas drawing system
- Canvas clear functionality
- Clean interactive UI

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- pyttsx3

---

## Model Details

- Custom dataset created using MediaPipe hand landmarks
- 42 input features (21 landmarks × x,y coordinates)
- Fully connected neural network:
  - Dense(128, ReLU)
  - Dense(64, ReLU)
  - Dense(32, ReLU)
  - Softmax output layer
- 30-class classification (A–Z + 4 words)
- ~96% test accuracy

---

## Controls

| Action | Control |
|--------|----------|
| Toggle Drawing Mode | Press `D` |
| Clear Canvas | Press `X` |
| Speak Current Word | Press `SPACE` |
| Clear Word Buffer | Press `C` |
| Exit | Press `ESC` |

---

