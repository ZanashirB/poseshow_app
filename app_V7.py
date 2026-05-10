import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# =========================================================
# 1. UI DESIGN & SETTINGS
# =========================================================
st.set_page_config(page_title="PoseShow Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; }
    .stButton>button { 
        background-color: #DFFF00; color: #000000; 
        border-radius: 12px; font-weight: 800; border: none;
        width: 100%; height: 3em; transition: 0.3s;
    }
    h1, h2, h3 { color: #DFFF00 !important; font-weight: 800; }
    img { border-radius: 15px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

st.title("PoseShow Pro")
st.write("Symmetrical Biometric Analysis Interface")

# =========================================================
# 2. CORE ENGINE SETUP
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
    # model_complexity=0 is the "Lite" version. 
    # It's faster for phones and avoids permission errors on the cloud server.
    return mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=0, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

# =========================================================
# 3. REAL-TIME STREAMING CLASS
# =========================================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = get_pose_engine()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Symmetrical Analysis Resize
        img = cv2.resize(img, (640, 480))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.engine.process(rgb)
        
        if results.pose_landmarks:
            # Draw the skeleton overlay
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(223, 255, 0), thickness=2)
            )
        
        # Return the processed frame to the user's browser
        return frame.from_ndarray(img, format="bgr24")

# =========================================================
# 4. NAVIGATION & MODES
# =========================================================
st.sidebar.markdown("## ⚙️ CONFIGURATION")
mode = st.sidebar.radio("Data Input", ["Real-time Webcam", "Image Analysis", "Video Analysis"])

# Simple configuration for browser security
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if mode == "Real-time Webcam":
    st.subheader("Live Biometric Stream")
    st.info("Click 'Start' to begin real-time analysis. Your browser will ask for camera permission.")
    
    webrtc_streamer(
        key="pose-stream",
        video_processor_factory=PoseProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

elif mode == "Image Analysis":
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
    if file:
        img_raw = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        engine = get_pose_engine()
        rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        results = engine.process(rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img_raw, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        st.image(img_raw, channels="BGR", use_container_width=True)

elif mode == "Video Analysis":
    file = st.file_uploader("Upload Video", type=['mp4','mov','avi'])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        v_proc = st.empty()
        engine = get_pose_engine()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
            res = engine.process(rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            v_proc.image(frame, channels="BGR", use_container_width=True)
        cap.release()