import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time

st.set_page_config(page_title="PoseShow Pro", layout="wide")

# CSS remains the same...
st.markdown("""<style>.stApp { background-color: #0E1117; color: #FFFFFF; }</style>""", unsafe_allow_html=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global dictionary to store the previous frame's landmarks for smoothing
last_pts = {}

@st.cache_resource
def get_pose_engine(comp):
    # Increased confidence thresholds to 0.7 to stop "hallucinating" low-confidence points
    return mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=comp, 
        smooth_landmarks=True, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )

def analyze_frame(frame, model_name, engine):
    global last_pts
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = engine.process(rgb)
    
    annotated = frame.copy()
    
    if results.pose_landmarks:
        h, w, _ = frame.shape
        # Extract points only if visibility is high
        curr_pts = {i: (int(lm.x * w), int(lm.y * h)) 
                    for i, lm in enumerate(results.pose_landmarks.landmark) 
                    if lm.visibility > 0.6}

        # Temporal Smoothing: 70% current, 30% previous
        alpha = 0.7
        for i, pos in curr_pts.items():
            if i in last_pts:
                curr_pts[i] = (int(alpha * pos[0] + (1 - alpha) * last_pts[i][0]),
                               int(alpha * pos[1] + (1 - alpha) * last_pts[i][1]))
            last_pts[i] = curr_pts[i]

        # Draw skeletons based on topology
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    return annotated

# App Layout
model_choice = st.sidebar.selectbox("Engine", ["MediaPipe", "MoveNet", "OpenPose"])
mode = st.sidebar.radio("Input", ["Video Analysis"])

engine = get_pose_engine(1) 

file = st.file_uploader("Upload Video", type=['mp4'])
if file:
    tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    # Optional: Confidence Chart for Thesis Data
    chart_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        proc = analyze_frame(frame, model_choice, engine)
        st_frame.image(proc, channels="BGR")
    cap.release()