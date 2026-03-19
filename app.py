import os
import platform
import tempfile
import threading
import time
from importlib import metadata
from typing import Any

import av
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.analytics import plot_angle_over_time, plot_angle_vs_stability, plot_stability_hist
from src.pose_analyzer import PoseAnalyzer
from src.state import CurlSessionState


class CurlVideoProcessor:
    def __init__(self) -> None:
        self.analyzer = PoseAnalyzer()
        self.state = CurlSessionState()
        self._lock = threading.Lock()
        self._latest_metrics: dict[str, Any] = {
            "angle": None,
            "movement": 0.0,
            "counter": 0,
            "stage": None,
            "feedback": "Waiting for pose...",
            "stability_feedback": "Elbow stability: Unknown",
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        annotated, metrics = self.analyzer.process_frame(frame_bgr=frame_bgr, state=self.state)
        with self._lock:
            self._latest_metrics = metrics
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "metrics": dict(self._latest_metrics),
                "angle_data": list(self.state.angle_data),
                "stability_data": list(self.state.stability_data),
                "reps_data": list(self.state.reps_data),
            }


def initialize_session_state() -> None:
    if "angle_data" not in st.session_state:
        st.session_state.angle_data = []
    if "stability_data" not in st.session_state:
        st.session_state.stability_data = []
    if "reps_data" not in st.session_state:
        st.session_state.reps_data = []
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {
            "angle": None,
            "movement": 0.0,
            "counter": 0,
            "stage": None,
            "feedback": "Waiting for pose...",
            "stability_feedback": "Elbow stability: Unknown",
        }


def update_session_from_snapshot(snapshot: dict[str, Any]) -> None:
    st.session_state.angle_data = snapshot["angle_data"]
    st.session_state.stability_data = snapshot["stability_data"]
    st.session_state.reps_data = snapshot["reps_data"]
    st.session_state.last_metrics = snapshot["metrics"]


def reset_session_data() -> None:
    st.session_state.angle_data = []
    st.session_state.stability_data = []
    st.session_state.reps_data = []
    st.session_state.last_metrics = {
        "angle": None,
        "movement": 0.0,
        "counter": 0,
        "stage": None,
        "feedback": "Waiting for pose...",
        "stability_feedback": "Elbow stability: Unknown",
    }


def render_metrics(metrics: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reps", metrics.get("counter") or 0)
    angle = metrics.get("angle")
    col2.metric("Elbow Angle", "-" if angle is None else f"{angle:.1f}°")
    col3.metric("Movement", f"{metrics.get('movement', 0.0):.4f}")
    col4.metric("Stage", metrics.get("stage") or "-")
    st.write(metrics.get("stability_feedback") or "")
    st.write(metrics.get("feedback") or "")


def _get_installed_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def render_startup_self_check() -> None:
    expected_versions = {
        "streamlit": "1.44.1",
        "streamlit-webrtc": "0.62.4",
        "mediapipe": "0.10.21",
        "opencv-python-headless": "4.11.0.86",
        "av": "14.2.0",
        "aiortc": "1.11.0",
        "aioice": "0.10.2",
    }
    rows: list[tuple[str, str | None, str]] = []
    has_drift = False

    for package_name, expected in expected_versions.items():
        installed = _get_installed_version(package_name)
        status = "✅"
        if installed is None:
            status = "❌ missing"
            has_drift = True
        elif installed != expected:
            status = "⚠️ drift"
            has_drift = True
        rows.append((package_name, installed, f"{status} (expected {expected})"))

    with st.expander("Startup self-check", expanded=False):
        st.write(f"Python: {platform.python_version()}")
        st.write(f"Platform: {platform.system()} {platform.release()}")
        st.table(
            {
                "package": [row[0] for row in rows],
                "installed": [row[1] if row[1] is not None else "-" for row in rows],
                "status": [row[2] for row in rows],
            }
        )

        if has_drift:
            st.warning("Environment drift detected. Rebuild/redeploy recommended.")
        else:
            st.success("Environment matches pinned versions.")


def process_uploaded_video(temp_path: str) -> None:
    cap = cv2.VideoCapture(temp_path)
    analyzer = PoseAnalyzer()
    state = CurlSessionState()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    progress = st.progress(0)
    preview = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = analyzer.process_frame(frame_bgr=frame, state=state)
        frame_index += 1

        if frame_index % 5 == 0:
            preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", caption="Processing uploaded video")

        if total_frames > 0:
            progress.progress(min(frame_index / total_frames, 1.0))

    cap.release()
    progress.progress(1.0)

    st.session_state.angle_data = list(state.angle_data)
    st.session_state.stability_data = list(state.stability_data)
    st.session_state.reps_data = list(state.reps_data)
    st.session_state.last_metrics = {
        "angle": state.last_angle,
        "movement": state.last_movement,
        "counter": state.counter,
        "stage": state.stage,
        "feedback": state.last_feedback,
        "stability_feedback": state.last_stability_feedback,
    }


def render_analytics() -> None:
    angle_data = st.session_state.angle_data
    stability_data = st.session_state.stability_data
    reps_data = st.session_state.reps_data

    if not angle_data or not stability_data:
        st.info("No analysis data yet. Use Live Camera or Upload Video first.")
        return

    st.subheader("Session Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Total Reps", st.session_state.last_metrics.get("counter") or 0)
    summary_col2.metric("Average Angle", f"{sum(angle_data) / len(angle_data):.1f}°")
    summary_col3.metric("Max Movement", f"{max(stability_data):.4f}")

    st.pyplot(plot_angle_over_time(angle_data))
    st.pyplot(plot_stability_hist(stability_data))
    st.pyplot(plot_angle_vs_stability(angle_data, stability_data))

    min_len = min(len(angle_data), len(stability_data), len(reps_data))
    export_df = pd.DataFrame(
        {
            "frame": list(range(1, min_len + 1)),
            "angle": angle_data[:min_len],
            "stability": stability_data[:min_len],
            "reps": reps_data[:min_len],
        }
    )

    st.download_button(
        "Download Session Data (CSV)",
        data=export_df.to_csv(index=False),
        file_name="bicep_curl_session.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Bicep Curl Analysis", layout="wide")
    initialize_session_state()

    st.title("Bicep Curl Analysis")
    st.caption("Real-time pose-based rep counting and form feedback")

    with st.sidebar:
        st.header("Controls")
        if st.button("Reset Session Data"):
            reset_session_data()
            st.success("Session data cleared")
        render_startup_self_check()

    live_tab, upload_tab, analytics_tab = st.tabs(["Live Camera", "Upload Video", "Analytics"])

    with live_tab:
        st.subheader("Live Webcam Analysis")
        st.write("Allow camera access in your browser, then start streaming.")

        webrtc_ctx = webrtc_streamer(
            key="bicep-curl-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=CurlVideoProcessor,
            async_processing=True,
        )

        metrics_placeholder = st.empty()
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            snapshot = webrtc_ctx.video_processor.get_snapshot()
            update_session_from_snapshot(snapshot)
            with metrics_placeholder.container():
                render_metrics(st.session_state.last_metrics)
            time.sleep(0.2)
            st.rerun()
        elif st.session_state.last_metrics:
            with metrics_placeholder.container():
                render_metrics(st.session_state.last_metrics)

    with upload_tab:
        st.subheader("Upload Video Analysis")
        uploaded_file = st.file_uploader("Upload MP4/MOV/AVI", type=["mp4", "mov", "avi"])

        if uploaded_file is not None and st.button("Process Uploaded Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            try:
                process_uploaded_video(temp_path=temp_path)
                st.success("Uploaded video processed successfully")
                render_metrics(st.session_state.last_metrics)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    with analytics_tab:
        st.subheader("Analytics")
        render_analytics()


if __name__ == "__main__":
    main()
