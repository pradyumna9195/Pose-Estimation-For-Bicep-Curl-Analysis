# Bicep Curl Analysis App

This project has been migrated from a Jupyter notebook into a deployable real-time Streamlit application using `streamlit-webrtc` for browser webcam support.

## Features

- Real-time webcam bicep curl analysis
- Elbow angle detection and curl stage tracking
- Repetition counting (`down -> up` transition)
- Elbow stability feedback based on movement threshold
- Upload-video fallback mode for non-WebRTC environments
- Analytics dashboard:
  - Angle over time
  - Stability histogram
  - Angle vs stability scatter plot
- CSV export of session data

## Tech Stack

- Streamlit
- streamlit-webrtc
- MediaPipe Pose
- OpenCV
- NumPy
- Matplotlib
- Pandas


## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Usage

### Live Camera

1. Open **Live Camera** tab.
2. Allow browser webcam permission.
3. Start the stream.
4. View real-time overlays: angle, stage, reps, and feedback.

### Upload Video

1. Open **Upload Video** tab.
2. Upload an `.mp4`, `.mov`, or `.avi` file.
3. Click **Process Uploaded Video**.

### Analytics

After a live session or upload run, open **Analytics** tab to view charts and download CSV session data.

## Deployment Notes

### Why `streamlit-webrtc`

Plain Streamlit server-side camera capture (`cv2.VideoCapture(0)`) does not access end-user webcams in hosted deployments.
`streamlit-webrtc` uses browser WebRTC and supports deployed camera usage.

### Requirements for Camera in Deployment

- HTTPS is required in most browsers for camera access.
- Users must allow webcam permission.
- STUN/TURN/network restrictions may affect stream startup in some networks.

## Troubleshooting

- **No camera prompt appears**
  - Ensure browser permissions are not blocked.
  - Use HTTPS in deployed environments.

- **No pose detected**
  - Ensure upper body is visible and lighting is adequate.
  - Keep camera at chest-to-head framing.

- **WebRTC fails to connect**
  - Try a different network/browser.
  - Use Upload Video fallback mode.

- **Low FPS / lag**
  - Close background apps using camera/CPU.
  - Use a lower-resolution camera input if available.

## Original Notebook

The original notebook implementation is preserved in `Pose_Bicep_Curl3.ipynb`.
