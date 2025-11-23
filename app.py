import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Cheating Detection", layout="wide")

st.title("Exam Cheating Detection – Live Webcam")
st.write("Detection will show whether student is **Cheating** or **Not Cheating**.")

run = st.checkbox("Start Detection")

FRAME_WINDOW = st.image([])

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

def detect_cheating(frame):
    """Simple logic: 
       If head rotates left/right or mouth opens → cheating
    """
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face:
        result = face.process(rgb)
        if not result.multi_face_landmarks:
            return frame, "No Face Detected"

        lm = result.multi_face_landmarks[0].landmark

        # Key landmarks
        left_eye = lm[33]      # Right eye
        right_eye = lm[263]    # Left eye
        mouth_top = lm[13]
        mouth_bottom = lm[14]

        # Eye horizontal distance
        eye_diff = abs(left_eye.x - right_eye.x)

        # Mouth opening value
        mouth_open = abs(mouth_top.y - mouth_bottom.y)

        # Thresholds
        cheating_eye = eye_diff < 0.21   # looking sideways
        cheating_mouth = mouth_open > 0.035  # opening mouth

        if cheating_eye or cheating_mouth:
            status = "CHEATING"
            color = (0, 0, 255)
        else:
            status = "NOT CHEATING"
            color = (0, 255, 0)

        # Put label
        cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return frame, status


camera = None

if run:
    camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera error: cannot read frame")
        break

    frame, result = detect_cheating(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)
