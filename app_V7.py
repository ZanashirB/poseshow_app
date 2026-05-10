import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# =========================================================
# 1.UI DESIGN (CSS)
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
    return mp_pose.Pose(
        static_image_mode=is_static, 
        model_complexity=comp, 
        smooth_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

def draw_movenet(frame, pts):
    line_color = (0, 255, 200) 
    face_links = [("nose","r_eye"), ("nose","l_eye"), ("r_eye","r_ear"), ("l_eye","l_ear")]
    body_links = [("r_sho","l_sho"), ("r_sho","r_hip"), ("l_sho","l_hip"), ("r_hip","l_hip"),
                  ("r_sho","r_elb"), ("r_elb","r_wri"), ("l_sho","l_elb"), ("l_elb","l_wri"),
                  ("r_hip","r_kne"), ("r_kne","r_ank"), ("l_hip","l_kne"), ("l_kne","l_ank")]
    for p1, p2 in face_links + body_links:
        if p1 in pts and p2 in pts:
            cv2.line(frame, pts[p1], pts[p2], line_color, 2, cv2.LINE_AA)
    return frame

def draw_openpose(frame, pts):
    line_color = (0, 0, 255) 
    if "r_sho" in pts and "l_sho" in pts:
        neck = (int((pts["r_sho"][0] + pts["l_sho"][0])/2), int((pts["r_sho"][1] + pts["l_sho"][1])/2))
        pts["neck"] = neck
    skeleton = [("nose","neck"), ("neck","r_sho"), ("neck","l_sho"), ("neck","r_hip"), ("neck","l_hip"),
                ("r_sho","r_elb"), ("r_elb","r_wri"), ("l_sho","l_elb"), ("l_elb","l_wri"),
                ("r_hip","r_kne"), ("r_kne","r_ank"), ("l_hip","l_kne"), ("l_kne","l_ank"),
                ("nose","r_eye"), ("r_eye","r_ear"), ("nose","l_eye"), ("l_eye","l_ear")]
    for p1, p2 in skeleton:
        if p1 in pts and p2 in pts:
            cv2.line(frame, pts[p1], pts[p2], line_color, 2, cv2.LINE_AA)
    return frame

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

        if "MediaPipe" in model_name:
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(223, 255, 0), thickness=2))
        elif "MoveNet" in model_name:
            annotated = draw_movenet(annotated, pts)
        else:
            annotated = draw_openpose(annotated, pts)
        
        for pt in pts.values():
            cv2.circle(annotated, pt, 4, (255, 255, 255), -1)

        score = np.mean([lm.visibility for lm in results.pose_landmarks.landmark]) * 100
        
    return frame, annotated, score, latency

# =========================================================
# 4. WEBRTC LIVE PROCESSOR (The Cloud Camera Fix)
# =========================================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self, model_name):
        self.model_name = model_name
        self.engine = get_pose_engine(0, False)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Reuse your reliable analysis logic
        _, proc, _, _ = analyze_frame(img, self.model_name, self.engine)
        return frame.from_ndarray(proc, format="bgr24")

# =========================================================
# 5. NAVIGATION
# =========================================================
st.sidebar.markdown("## ⚙️ CONFIGURATION")
model_choice = st.sidebar.selectbox("Analysis Engine", ["MediaPipe (33 pts)", "MoveNet (17 pts)", "OpenPose (18 pts)"])
mode = st.sidebar.radio("Data Input", ["Real-time Webcam", "Image Analysis", "Video Analysis"])

# =========================================================
# 6. EXECUTION MODES
# =========================================================

if "Webcam" in mode:
    st.info("Ensure you are using HTTPS. Click START to begin live tracking.")
    # STUN server helps camera connect through different network types
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    webrtc_streamer(
        key="pose-stream",
        video_processor_factory=lambda: PoseProcessor(model_choice),
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

elif "Image" in mode:
    engine = get_pose_engine(2, True)
    file = st.file_uploader("Upload Image Asset", type=['jpg','png','jpeg'])
    if file:
        img_raw = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        orig_resized, proc, score, lat = analyze_frame(img_raw, model_choice, engine)
        c1, c2 = st.columns(2)
        c1.image(orig_resized, channels="BGR", use_container_width=True, caption="Raw Input")
        c2.image(proc, channels="BGR", use_container_width=True, caption=f"{model_choice} Output")
        st.success(f"{int(score)}% Confidence | {int(lat)}ms")

elif "Video" in mode:
    engine = get_pose_engine(2, False)
    file = st.file_uploader("Upload Video File", type=['mp4','mov','avi'])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        c1, c2 = st.columns(2)
        v_orig = c1.empty(); v_proc = c2.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            orig_resized, proc, score, lat = analyze_frame(frame, model_choice, engine)
            v_orig.image(orig_resized, channels="BGR", use_container_width=True)
            v_proc.image(proc, channels="BGR", use_container_width=True)
        cap.release()