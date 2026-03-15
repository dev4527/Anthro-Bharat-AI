import os
import subprocess
import sys

# --- RUNTIME INSTALLATION HACK ---
# Streamlit Cloud par MediaPipe install karne ka sabse foolproof tareeka
try:
    import mediapipe as mp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe==0.10.14"])
    import mediapipe as mp

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import pandas as pd
from datetime import datetime

# --- Setup MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- Helper Functions ---
def calculate_angle(a, b, c):
    """Calculate angle between three points (a, b, c) with b as the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_vertical_angle(top_point, bottom_point):
    """Calculate angle of a body part with the true vertical line."""
    dx = top_point[0] - bottom_point[0]
    dy = top_point[1] - bottom_point[1]
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Anthro-Bharat AI | Mining Ergonomics", layout="wide", page_icon="⛏️")

# --- Sidebar Branding ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    st.info("Upload clear images or use the camera for real-time mining posture analysis.")
    st.markdown("### Safety Guidelines")
    st.markdown("- **Back Bending > 20°**: High Risk\n- **Knee Angle < 90°**: Joint Stress\n- **Arms > 45°**: Shoulder Strain")
    st.markdown("---")
    st.markdown("👨‍💻 **Developed by Vivek**")
    st.markdown("Built with Streamlit & MediaPipe")

st.title("⛏️ Anthro-Bharat AI: Advanced Mining Ergonomics")
st.markdown("**Real-time posture risk assessment for mining operations.**")

# --- Input Selection ---
input_mode = st.radio("Select Image Source:", ("📸 Live Camera", "📁 Upload Photo"), horizontal=True)

image_np = None

if input_mode == "📸 Live Camera":
    camera_photo = st.camera_input("Take a picture of the worker")
    if camera_photo is not None:
        image_pil = Image.open(camera_photo)
        image_np = np.array(image_pil)
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil)

# --- Process Image ---
if image_np is not None:
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    image_rgb = image_np 
    h, w, _ = image_rgb.shape
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        annotated_image = image_rgb.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        
        def get_coords(idx):
            return [landmarks[idx].x * w, landmarks[idx].y * h]

        try:
            l_shoulder, l_hip = get_coords(11), get_coords(23)
            l_knee, l_ankle = get_coords(25), get_coords(27)
            l_elbow = get_coords(13)

            back_angle = calculate_vertical_angle(l_shoulder, l_hip)
            knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            arm_angle = calculate_vertical_angle(l_elbow, l_shoulder)

            col1, col2 = st.columns([1.5, 1])

            with col1:
                st.image(annotated_image, caption="AI Posture Analysis", use_container_width=True)

            with col2:
                st.subheader("📊 Ergonomic Risk Report")
                
                # Metrics
                st.metric("Back Bending", f"{back_angle:.1f}°")
                if back_angle < 20: st.success("🟢 Safe")
                else: st.error("🔴 High Risk")

                st.metric("Knee Flexion", f"{knee_angle:.1f}°")
                if 90 <= knee_angle <= 120: st.success("🟢 Safe")
                else: st.error("🔴 High Risk")

                st.metric("Arm Elevation", f"{arm_angle:.1f}°")
                if arm_angle < 45: st.success("🟢 Safe")
                else: st.warning("🟡 High Risk")

                # Export
                st.markdown("---")
                report_data = {
                    "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Back Angle": [round(back_angle, 2)],
                    "Knee Angle": [round(knee_angle, 2)],
                    "Overall Risk": ["High" if back_angle > 20 or knee_angle < 90 else "Safe"]
                }
                df = pd.DataFrame(report_data)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Report", data=csv, file_name="Ergo_Report.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
    else:
        st.error("⚠️ No human detected.")