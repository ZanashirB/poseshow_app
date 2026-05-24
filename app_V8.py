import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Page Config
st.set_page_config(page_title="PoseShow Pro", layout="wide")
st.title("PoseShow Pro - Integrated Prototype")

# Initialize Engine (cached to prevent reload loops)
@st.cache_resource
def get_pose_engine():
    return mp.solutions.pose.Pose(
        static_image_mode=False, 
        model_complexity=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

engine = get_pose_engine()

# Processing Function
def analyze_frame(frame, engine):
    # Resize to reduce processing load
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = engine.process(rgb)
    
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS
        )
    return frame

# UI Navigation
mode = st.sidebar.radio("Input Mode", ["Real-time Webcam", "Image Analysis"])

# --- REAL-TIME WEBCAM ---
if mode == "Real-time Webcam":
    if st.button("Start Camera"):
        cam = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        stop_btn = st.button("Stop Camera")
        
        while cam.isOpened() and not stop_btn:
            ret, frame = cam.read()
            if not ret: break
            frame = analyze_frame(frame, engine)
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
        cam.release()

# --- IMAGE ANALYSIS ---
elif mode == "Image Analysis":
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        proc = analyze_frame(img, engine)
        st.image(proc, channels="BGR", use_container_width=True)