import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# =========================================================
# 1. UI DESIGN & SIDEBAR
# =========================================================
st.set_page_config(page_title="PoseShow Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; }
    .stButton>button { 
        background-color: #DFFF00; color: #000000; 
        border-radius: 12px; font-weight: 800; border: none;
    }
    h1, h2, h3 { color: #DFFF00 !important; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

st.title("PoseShow Pro")

st.sidebar.markdown("## ⚙️ CONFIGURATION")
model_choice = st.sidebar.selectbox("Analysis Engine", ["MediaPipe (33 pts)", "MoveNet (17 pts)", "OpenPose (18 pts)"])
mode = st.sidebar.radio("Data Input", ["Real-time Webcam", "Image Analysis", "Video Analysis"])

# =========================================================
# 2. CORE ENGINE & DRAWING LOGIC
# =========================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def get_pose_engine(is_static=False):
    return mp_pose.Pose(
        static_image_mode=is_static, 
        model_complexity=0, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

def draw_skeleton(frame, landmarks, style):
    if not landmarks: return frame
    
    if "MediaPipe" in style:
        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
                                 mp_drawing.DrawingSpec(color=(223, 255, 0), thickness=2))
    elif "MoveNet" in style:
        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
                                 mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))
    else: # OpenPose
        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
                                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
    return frame

# =========================================================
# 3. WEBRTC LIVE STREAMING
# =========================================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self, model_style):
        self.engine = get_pose_engine(is_static=False)
        self.model_style = model_style

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.engine.process(rgb)
        if results.pose_landmarks:
            img = draw_skeleton(img, results.pose_landmarks, self.model_style)
        return frame.from_ndarray(img, format="bgr24")

# =========================================================
# 4. EXECUTION MODES
# =========================================================
# STUN Servers help the camera work on different networks/phones
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if mode == "Real-time Webcam":
    st.subheader("Live Biometric Stream")
    webrtc_streamer(
        key="pose-stream",
        video_processor_factory=lambda: PoseProcessor(model_choice),
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

elif mode == "Image Analysis":
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        engine = get_pose_engine(is_static=True)
        results = engine.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            img = draw_skeleton(img, results.pose_landmarks, model_choice)
        st.image(img, channels="BGR", use_container_width=True)

elif mode == "Video Analysis":
    file = st.file_uploader("Upload Video", type=['mp4','mov','avi'])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        engine = get_pose_engine(is_static=False)
        v_place = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = engine.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                frame = draw_skeleton(frame, results.pose_landmarks, model_choice)
            v_place.image(frame, channels="BGR", use_container_width=True)
        cap.release()