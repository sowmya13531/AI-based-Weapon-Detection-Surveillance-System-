import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import os

# --- Config ---
st.set_page_config(page_title="AI Weapon Detection Survillience", layout="wide")
MODEL_PATH = "best (3).pt"  # <-- Replace with your custom model path
SAVE_FRAMES_DIR = "saved_frames"
LOG_CSV = "detection_log.csv"
os.makedirs(SAVE_FRAMES_DIR, exist_ok=True)

# --- Load Model ---
model = YOLO(MODEL_PATH)
# --- Keywords for Weapon Detection ---
weapon_keywords = ["knife", "gun", "pistol", "rifle", "weapon"]
class_map = {
    "handgun": "pistol",
    "sharp_object": "knife",
    "assault_rifle": "rifle"
}

# --- Session State ---
if "log_rows" not in st.session_state:
    st.session_state.log_rows = []

def log_detection(timestamp, source, class_name, confidence, frame_path=None):
    row = {
        "timestamp": timestamp,
        "source": source,
        "class": class_name,
        "confidence": confidence,
        "frame_path": frame_path or ""
    }
    st.session_state.log_rows.append(row)
    df = pd.DataFrame(st.session_state.log_rows)
    df.to_csv(LOG_CSV, index=False)

# --- Sidebar UI ---
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Input", ("Webcam", "Upload Image", "Upload Video", "Upload CSV"))
confidence_thr = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# --- Title ---
st.title("🔫 AI Weapon Detection Survillience System")
frame_placeholder = st.empty()

# --- Webcam Mode ---
if mode == "Webcam":
    st.sidebar.write("📷 Start Webcam Feed")
    run_webcam = st.sidebar.button("Start Webcam")
    stop_webcam = st.sidebar.button("Stop Webcam")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated = results.plot()
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls, str(cls)).lower()
                name = class_map.get(name, name)

                if conf >= confidence_thr and any(w in name for w in weapon_keywords):
                    ts = datetime.now().isoformat()
                    fname = f"{name}_{ts.replace(':', '_').replace('.', '_')}.jpg"
                    fpath = os.path.join(SAVE_FRAMES_DIR, fname)
                    cv2.imwrite(fpath, frame)
                    log_detection(ts, "webcam", name, conf, fpath)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_column_width=True)
            if stop_webcam:
                cap.release()
                break

# --- Image Upload Mode ---
elif mode == "Upload Image":
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = np.array(Image.open(img_file).convert("RGB"))
        results = model(img)[0]
        annotated = results.plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Detected Image", use_column_width=True)

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names.get(cls, str(cls)).lower()
            name = class_map.get(name, name)

            if conf >= confidence_thr and any(w in name for w in weapon_keywords):
                ts = datetime.now().isoformat()
                fname = f"{name}_{ts.replace(':', '_').replace('.', '_')}.jpg"
                fpath = os.path.join(SAVE_FRAMES_DIR, fname)
                cv2.imwrite(fpath, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                log_detection(ts, "image_upload", name, conf, fpath)

# --- Video Upload Mode ---
elif mode == "Upload Video":
    vid_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if vid_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(vid_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output_annotated.mp4", fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated = results.plot()

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls, str(cls)).lower()
                name = class_map.get(name, name)

                if conf >= confidence_thr and any(w in name for w in weapon_keywords):
                    ts = datetime.now().isoformat()
                    log_detection(ts, "video_upload", name, conf)

            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_column_width=True)
            out.write(annotated)

        cap.release()
        out.release()
        st.success("Video processing complete.")
        st.video("output_annotated.mp4")

# --- CSV Upload Mode ---
elif mode == "Upload CSV":
    csv_file = st.file_uploader("Upload detection CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.subheader("📊 CSV Detection Data")
        st.dataframe(df)
        if "image_path" in df.columns:
            images = df["image_path"].unique()
            selected = st.selectbox("Select Image to View Detections", images)
            filtered = df[df["image_path"] == selected]
            image = cv2.imread(selected)
            if image is not None:
                for _, row in filtered.iterrows():
                    x1, y1 = int(row["x_min"]), int(row["y_min"])
                    x2, y2 = int(row["x_max"]), int(row["y_max"])
                    label = f'{row["class"]} ({row["confidence"]:.2f})'
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            else:
                st.warning("⚠️ Image not found in the given path.")

# --- Detection Log ---
st.sidebar.subheader("📝 Detection Log")
if st.session_state.log_rows:
    df_log = pd.DataFrame(st.session_state.log_rows)
    st.sidebar.dataframe(df_log)
    st.sidebar.download_button("Download Detection Log CSV",
                               data=df_log.to_csv(index=False),
                               file_name="detection_log.csv",
                               mime="text/csv")
