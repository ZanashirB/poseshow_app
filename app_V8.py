import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
import os

# =========================================================
# 1. UI DESIGN (CSS)
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
    .stButton>button:hover { background-color: #BAE600; color: #000; }
    .stMetric { 
        background-color: #161B22; padding: 20px; 
        border-radius: 15px; border: 1px solid #30363D; 
    }
    div[data-testid="stMetricValue"] { color: #DFFF00 !important; font-family: 'Courier New'; }
    h1, h2, h3 { color: #DFFF00 !important; font-weight: 800; }
    img { border-radius: 15px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

st.title("PoseShow Pro")
st.write("Symmetrical Biometric Analysis Interface")

# =========================================================
# 2. DEFINITIONS
# =========================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

JOINTS = {
    "nose": 0, "r_eye": 5, "l_eye": 2, "r_ear": 8, "l_ear": 7,
    "r_sho": 12, "l_sho": 11, "r_elb": 14, "l_elb": 13, "r_wri": 16, "l_wri": 15,
    "r_hip": 24, "l_hip": 23, "r_kne": 26, "l_kne": 25, "r_ank": 28, "l_ank": 27
}

# =========================================================
# 3. CORE ENGINE
# =========================================================
@st.cache_resource
def get_pose_engine(comp, is_static):
    # Writable directory for cloud deployment
    os.environ['MEDIAPIPE_MODEL_PATH'] = '/tmp'
    return mp_pose.Pose(
        static_image_mode=is_static, 
        model_complexity=comp, 
        smooth_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

def draw_movenet(frame, pts):
    line_color = (0, 255, 200) 
    links = [("nose","r_eye"), ("nose","l_eye"), ("r_sho","l_sho"), ("r_sho","r_hip"), 
             ("l_sho","l_hip"), ("r_hip","l_hip"), ("r_sho","r_elb"), ("r_elb","r_wri"), 
             ("l_sho","l_elb"), ("l_elb","l_wri"), ("r_hip","r_kne"), ("r_kne","r_ank"), 
             ("l_hip","l_kne"), ("l_kne","l_ank")]
    for p1, p2 in links:
        if p1 in pts and p2 in pts: cv2.line(frame, pts[p1], pts[p2], line_color, 2, cv2.LINE_AA)
    return frame

def draw_openpose(frame, pts):
    line_color = (0, 0, 255) 
    if "r_sho" in pts and "l_sho" in pts:
        neck = (int((pts["r_sho"][0] + pts["l_sho"][0])/2), int((pts["r_sho"][1] + pts["l_sho"][1])/2))
        pts["neck"] = neck
    skeleton = [("nose","neck"), ("neck","r_sho"), ("neck","l_sho"), ("r_sho","r_elb"), ("r_elb","r_wri"),
                ("l_sho","l_elb"), ("l_elb","l_wri"), ("neck","r_hip"), ("neck","l_hip"),
                ("r_hip","r_kne"), ("r_kne","r_ank"), ("l_hip","l_kne"), ("l_kne","l_ank")]
    for p1, p2 in skeleton:
        if p1 in pts and p2 in pts: cv2.line(frame, pts[p1], pts[p2], line_color, 2, cv2.LINE_AA)
    return frame

def analyze_frame(frame, model_name, engine, target_size=(640, 480)):
    frame = cv2.resize(frame, target_size)
    results = engine.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    start = time.time()
    annotated = frame.copy()
    score = 0
    if results.pose_landmarks:
        h, w, _ = frame.shape
        pts = {k: (int(landmarks.landmark[v].x * w), int(landmarks.landmark[v].y * h)) 
               for k, v in JOINTS.items() for landmarks in [results.pose_landmarks] 
               if landmarks.landmark[v].visibility > 0.5}
        if "MediaPipe" in model_name:
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        elif "MoveNet" in model_name: annotated = draw_movenet(annotated, pts)
        else: annotated = draw_openpose(annotated, pts)
        score = np.mean([lm.visibility for lm in results.pose_landmarks.landmark]) * 100
    return annotated, score, (time.time() - start) * 1000

# =========================================================
# 4. EXECUTION
# =========================================================
model_choice = st.sidebar.selectbox("Engine", ["MediaPipe", "MoveNet", "OpenPose"])
mode = st.sidebar.radio("Input", ["Real-time Webcam", "Image", "Video"])
engine = get_pose_engine(0 if mode == "Real-time Webcam" else 2, mode == "Image")

if "Webcam" in mode:
    m1, m2, m3 = st.columns(3); q_met, l_met, f_met = m1.empty(), m2.empty(), m3.empty()
    video_panel = st.empty(); prev_t = 0
    if st.button("START"):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            curr_t = time.time()
            proc, score, lat = analyze_frame(frame, model_choice, engine)
            q_met.metric("CONFIDENCE", f"{int(score)}%"); l_met.metric("LATENCY", f"{int(lat)}ms")
            f_met.metric("FPS", f"{int(1/(curr_t - prev_t) if prev_t > 0 else 0)}")
            video_panel.image(proc, channels="BGR", use_container_width=True)
            prev_t = curr_t
        cap.release()
elif "Image" in mode:
    file = st.file_uploader("Upload", type=['jpg','png'])
    if file:
        proc, score, lat = analyze_frame(cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1), model_choice, engine)
        st.image(proc, channels="BGR", use_container_width=True)
        st.success(f"Confidence: {int(score)}% | Latency: {int(lat)}ms")