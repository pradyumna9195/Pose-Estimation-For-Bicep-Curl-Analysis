from src.config import ANGLE_DOWN_THRESHOLD, ANGLE_UP_THRESHOLD, ELBOW_MOVEMENT_THRESHOLD


def generate_feedback(angle: float, movement: float) -> tuple[str, str]:
    elbow_feedback = "Elbow stability: Good"
    if movement > ELBOW_MOVEMENT_THRESHOLD:
        elbow_feedback = "Elbow is moving too much! Please stabilize."
        return elbow_feedback, "Form: Not good"

    if angle > ANGLE_DOWN_THRESHOLD:
        return elbow_feedback, "Curl up (bring the weight up)."
    if angle < ANGLE_UP_THRESHOLD:
        return elbow_feedback, "Go down (lower the weight)."
    return elbow_feedback, "Good form! Keep going."
