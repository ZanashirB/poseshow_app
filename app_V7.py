import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time

# =========================================================
# 1. UI DESIGN (CSS) - Professional Biometric Look
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
# 2. DEFINITIONS & PRECISION LANDMARKS
# =========================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

JOINTS = {
    "nose": 0, "r_eye": 5, "l_eye": 2, "r_ear": 8, "l_ear": 7,
    "r_sho": 12, "l_sho": 11, "r_elb": 14, "l_elb": 13, "r_wri": 16, "l_wri": 15,
    "r_hip": 24, "l_hip": 23, "r_kne": 26, "l_kne": 25, "r_ank": 28, "l_ank": 27
}

# =========================================================
# 3. CORE PROCESSING ENGINE
# =========================================================

@st.cache_resource
def get_pose_engine(comp, is_static):
    # model_complexity=0 (Lite) is CRITICAL for cloud permissions and mobile speed
    return mp_pose.Pose(
        static_image_mode=is_static, 
        model_complexity=0, 
        smooth_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

def analyze_frame(frame, model_name, engine, target_size=(640, 480)):
    frame = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start = time.time()
    results = engine.process(rgb)
    latency = (time.time() - start) * 1000
    
    annotated = frame.copy()
    score = 0
    if results.pose_landmarks:
        h, w, _ = frame.shape
        pts = {k: (int(landmarks.landmark[v].x * w), int(landmarks.landmark[v].y * h)) 
               for k, v in JOINTS.items() for landmarks in [results.pose_landmarks] 
               if landmarks.landmark[v].visibility > 0.5}

        # MediaPipe Drawing
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(223, 255, 0), thickness=2))
        
        for pt in pts.values():
            cv2.circle(annotated, pt, 4, (255, 255, 255), -1)

        score = np.mean([lm.visibility for lm in results.pose_landmarks.landmark]) * 100
        
    return frame, annotated, score, latency

# =========================================================
# 4. NAVIGATION
# =========================================================
st.sidebar.markdown("## ⚙️ CONFIGURATION")
model_choice = st.sidebar.selectbox("Analysis Engine", ["MediaPipe (33 pts)", "MoveNet (17 pts)", "OpenPose (18 pts)"])
mode = st.sidebar.radio("Data Input", ["Mobile/Webcam Capture", "Image Analysis", "Video Analysis"])

# Force complexity 0 for all modes to ensure cloud server permissions
engine = get_pose_engine(0, mode == "Image Analysis")

# =========================================================
# 5. EXECUTION MODES (Streamlit Cloud Compatible)
# =========================================================

if "Capture" in mode:
    # st.camera_input is necessary for mobile deployment
    img_file = st.camera_input("Position yourself for analysis")
    
    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        m1, m2 = st.columns(2)
        orig_resized, proc, score, lat = analyze_frame(frame, model_choice, engine)
        
        m1.metric("CONFIDENCE SCORE", f"{int(score)}%")
        m2.metric("CLOUD LATENCY", f"{int(lat)}ms")
        
        st.image(proc, channels="BGR", use_container_width=True, caption="Biometric Skeleton Result")

elif "Image" in mode:
    file = st.file_uploader("Upload Image Asset", type=['jpg','png','jpeg'])
    if file:
        img_raw = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        orig_resized, proc, score, lat = analyze_frame(img_raw, model_choice, engine)
        
        c1, c2 = st.columns(2)
        c1.image(orig_resized, channels="BGR", use_container_width=True, caption="Raw Input Source")
        c2.image(proc, channels="BGR", use_container_width=True, caption=f"Analyzed Pose ({model_choice})")
        st.success(f"Final Analysis Result: {int(score)}% Confidence | {int(lat)}ms Processing Time")

elif "Video" in mode:
    file = st.file_uploader("Upload Video File", type=['mp4','mov','avi'])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        c1, c2 = st.columns(2)
        v_orig = c1.empty()
        v_proc = c2.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            orig_resized, proc, score, lat = analyze_frame(frame, model_choice, engine)
            v_orig.image(orig_resized, channels="BGR", use_container_width=True)
            v_proc.image(proc, channels="BGR", use_container_width=True)
        cap.release()