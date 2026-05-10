import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# =========================================================
# 1. UI & SIDEBAR
# =========================================================
st.set_page_config(page_title="PoseShow Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    h1, h2, h3 { color: #DFFF00 !important; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

st.title("PoseShow Pro")

st.sidebar.markdown("## ⚙️ CONFIGURATION")
model_choice = st.sidebar.selectbox("Analysis Engine", ["MediaPipe (33 pts)", "MoveNet (17 pts)", "OpenPose (18 pts)"])
mode = st.sidebar.radio("Data Input", ["Real-time Webcam", "Image Analysis"])

# =========================================================
# 2. CORE ENGINE & DRAWING LOGIC
# =========================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

JOINTS = {
    "nose": 0, "r_eye": 5, "l_eye": 2, "r_ear": 8, "l_ear": 7,
    "r_sho": 12, "l_sho": 11, "r_elb": 14, "l_elb": 13, "r_wri": 16, "l_wri": 15,
    "r_hip": 24, "l_hip": 23, "r_kne": 26, "l_kne": 25, "r_ank": 28, "l_ank": 27
}

@st.cache_resource
def get_pose_engine():
    return mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

def draw_custom_skeleton(frame, landmarks, style):
    h, w, _ = frame.shape
    pts = {k: (int(landmarks.landmark[v].x * w), int(landmarks.landmark[v].y * h)) 
           for k, v in JOINTS.items() if landmarks.landmark[v].visibility > 0.5}

    if "MediaPipe" in style:
        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
                                 mp_drawing.DrawingSpec(color=(223, 255, 0), thickness=2))
    elif "MoveNet" in style:
        # Simplied MoveNet style (Cyan)
        for p1, p2 in mp_pose.POSE_CONNECTIONS:
            idx1, idx2 = p1, p2
            lm1, lm2 = landmarks.landmark[idx1], landmarks.landmark[idx2]
            if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                cv2.line(frame, (int(lm1.x*w), int(lm1.y*h)), (int(lm2.x*w), int(lm2.y*h)), (255, 255, 0), 2)
    else: # OpenPose style (Red)
        for p1, p2 in mp_pose.POSE_CONNECTIONS:
            lm1, lm2 = landmarks.landmark[p1], landmarks.landmark[p2]
            if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                cv2.line(frame, (int(lm1.x*w), int(lm1.y*h)), (int(lm2.x*w), int(lm2.y*h)), (0, 0, 255), 2)
    return frame

# =========================================================
# 3. WEBRTC PROCESSOR
# =========================================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self, model_style):
        self.engine = get_pose_engine()
        self.model_style = model_style

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror for natural feel
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.engine.process(rgb)
        
        if results.pose_landmarks:
            img = draw_custom_skeleton(img, results.pose_landmarks, self.model_style)
        
        return frame.from_ndarray(img, format="bgr24")

# =========================================================
# 4. EXECUTION
# =========================================================
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if mode == "Real-time Webcam":
    # Passing the model_choice into the processor
    webrtc_streamer(
        key="pose-stream",
        video_processor_factory=lambda: PoseProcessor(model_choice),
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )