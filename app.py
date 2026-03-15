import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
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
    """Calculate angle of a body part with the true vertical line (e.g., for back bending)."""
    # Using atan2 to find angle with vertical (y-axis)
    dx = top_point[0] - bottom_point[0]
    dy = top_point[1] - bottom_point[1]
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Anthro-Bharat AI | Mining Ergonomics", layout="wide", page_icon="⛏️")

# --- NEW FEATURE: Sidebar Branding ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    st.info("Upload clear images or use the camera for real-time mining posture analysis.")
    st.markdown("### Safety Guidelines")
    st.markdown("- **Back Bending > 20°** is High Risk\n- **Knee Angle < 90°** causes joint stress\n- **Arms > 45°** elevation causes shoulder strain")
    st.markdown("---")
    st.markdown("👨‍💻 **Developed by Vivek**")
    st.markdown("Built with Streamlit & MediaPipe")

st.title("⛏️ Anthro-Bharat AI: Advanced Mining Ergonomics")
st.markdown("**Real-time posture risk assessment for heavy machinery operators and underground miners.**")

# --- Input Selection: Camera vs Upload ---
input_mode = st.radio("Select Image Source:", ("📸 Live Camera", "📁 Upload Photo"), horizontal=True)

image_np = None

if input_mode == "📸 Live Camera":
    camera_photo = st.camera_input("Take a picture of the worker")
    if camera_photo is not None:
        image_pil = Image.open(camera_photo)
        image_np = np.array(image_pil)
else:
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil)

# --- Process Image ---
if image_np is not None:
    # Handle RGBA to RGB conversion
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    elif len(image_np.shape) == 2: 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
    image_rgb = image_np 
    h, w, _ = image_rgb.shape

    # MediaPipe Processing
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        annotated_image = image_rgb.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        
        def get_coords(idx):
            return [landmarks[idx].x * w, landmarks[idx].y * h]

        try:
            # Extract key points (Left side for standard lateral view analysis)
            l_shoulder = get_coords(11)
            l_hip = get_coords(23)
            l_knee = get_coords(25)
            l_ankle = get_coords(27)
            l_elbow = get_coords(13)

            # --- Advanced Mining Ergonomic Calculations ---
            back_angle = calculate_vertical_angle(l_shoulder, l_hip)
            knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            arm_angle = calculate_vertical_angle(l_elbow, l_shoulder)

            # --- UI Display ---
            col1, col2 = st.columns([1.5, 1])

            with col1:
                st.image(annotated_image, caption="AI Posture Skeleton Analysis", use_container_width=True)

            with col2:
                st.subheader("📊 Ergonomic Risk Report")
                
                # Metric 1: Back Posture
                st.markdown("### 1. Trunk/Back Posture")
                st.metric("Back Bending Angle", f"{back_angle:.1f}°")
                if back_angle < 10:
                    st.success("🟢 Safe: Straight Back")
                elif back_angle < 20:
                    st.warning("🟡 Medium Risk: Slight Bending")
                else:
                    st.error("🔴 High Risk: Severe Back Stress")

                st.markdown("---")

                # Metric 2: Seating/Knee Posture
                st.markdown("### 2. Lower Body (Dumper Operator Seating)")
                st.metric("Knee Flexion Angle", f"{knee_angle:.1f}°")
                if 90 <= knee_angle <= 120:
                    st.success("🟢 Safe: Optimal Seating Posture")
                else:
                    st.error("🔴 High Risk: Cramped or Overextended Legs")

                st.markdown("---")
                
                # Metric 3: Arm Reaching
                st.markdown("### 3. Arm Elevation (Drilling/Controls)")
                st.metric("Shoulder-Arm Angle", f"{arm_angle:.1f}°")
                if arm_angle < 45:
                    st.success("🟢 Safe: Arms in optimal working zone")
                else:
                    st.warning("🟡 High Risk: Reaching too high")

                # --- NEW FEATURE: Data Export ---
                st.markdown("---")
                st.subheader("💾 Export Data")
                
                # Logic to decide overall risk for the report
                overall_risk = "High" if (back_angle > 20 or knee_angle < 90 or arm_angle > 45) else "Safe"

                # Create DataFrame
                report_data = {
                    "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Worker ID": ["Operator_01"],
                    "Back Bending Angle": [round(back_angle, 2)],
                    "Knee Flexion Angle": [round(knee_angle, 2)],
                    "Arm Elevation Angle": [round(arm_angle, 2)],
                    "Overall Risk": [overall_risk]
                }
                df = pd.DataFrame(report_data)
                
                # Generate CSV
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Audit Report (CSV)",
                    data=csv,
                    file_name=f"Mining_Ergo_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error calculating metrics. Please ensure the full body is visible. Details: {e}")
    else:
        st.error("⚠️ No human detected. Please adjust the camera or upload a clear photo.")