from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    ANGLE_DOWN_THRESHOLD,
    ANGLE_UP_THRESHOLD,
    BAD_COLOR,
    FRAME_TEXT_COLOR,
    GOOD_COLOR,
    INFO_COLOR,
    MAX_HISTORY_POINTS,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
)
from src.feedback import generate_feedback
from src.state import CurlSessionState


class PoseAnalyzer:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
        )

    @staticmethod
    def calculate_angle(
        shoulder: Tuple[float, float], elbow: Tuple[float, float], wrist: Tuple[float, float]
    ) -> float:
        angle = np.degrees(
            np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0])
            - np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
        )
        angle = float(np.abs(angle))
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    @staticmethod
    def calculate_movement(
        elbow_position: Tuple[float, float], prev_elbow_position: Optional[Tuple[float, float]]
    ) -> float:
        if prev_elbow_position is None:
            return 0.0
        return float(np.linalg.norm(np.array(elbow_position) - np.array(prev_elbow_position)))

    def process_frame(self, frame_bgr: np.ndarray, state: CurlSessionState) -> tuple[np.ndarray, Dict[str, Any]]:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            cv2.putText(
                image,
                "No pose detected. Ensure your upper body is visible.",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                BAD_COLOR,
                2,
                cv2.LINE_AA,
            )
            metrics = {
                "angle": state.last_angle,
                "movement": state.last_movement,
                "counter": state.counter,
                "stage": state.stage,
                "feedback": "No pose detected",
                "stability_feedback": state.last_stability_feedback,
            }
            return image, metrics

        landmarks = results.pose_landmarks.landmark
        elbow = (
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        )
        shoulder = (
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        )
        wrist = (
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        )

        angle = self.calculate_angle(shoulder=shoulder, elbow=elbow, wrist=wrist)
        movement = self.calculate_movement(elbow_position=elbow, prev_elbow_position=state.prev_elbow_position)
        stability_feedback, angle_feedback = generate_feedback(angle=angle, movement=movement)

        if angle > ANGLE_DOWN_THRESHOLD:
            state.stage = "down"
        if angle < ANGLE_UP_THRESHOLD and state.stage == "down":
            state.stage = "up"
            state.counter += 1

        state.prev_elbow_position = elbow
        state.last_angle = angle
        state.last_movement = movement
        state.last_feedback = angle_feedback
        state.last_stability_feedback = stability_feedback

        state.angle_data.append(angle)
        state.stability_data.append(movement)
        state.reps_data.append(state.counter)

        if len(state.angle_data) > MAX_HISTORY_POINTS:
            state.angle_data = state.angle_data[-MAX_HISTORY_POINTS:]
            state.stability_data = state.stability_data[-MAX_HISTORY_POINTS:]
            state.reps_data = state.reps_data[-MAX_HISTORY_POINTS:]

        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        cv2.putText(image, f"Elbow Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, FRAME_TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            stability_feedback,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            BAD_COLOR if "moving too much" in stability_feedback else GOOD_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(image, angle_feedback, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, INFO_COLOR, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"Stage: {state.stage}",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            FRAME_TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Reps: {state.counter}",
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            FRAME_TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        metrics = {
            "angle": angle,
            "movement": movement,
            "counter": state.counter,
            "stage": state.stage,
            "feedback": angle_feedback,
            "stability_feedback": stability_feedback,
        }
        return image, metrics
