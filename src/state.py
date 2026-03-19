from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CurlSessionState:
    counter: int = 0
    stage: Optional[str] = None
    prev_elbow_position: Optional[Tuple[float, float]] = None
    angle_data: List[float] = field(default_factory=list)
    stability_data: List[float] = field(default_factory=list)
    reps_data: List[int] = field(default_factory=list)
    last_angle: Optional[float] = None
    last_movement: float = 0.0
    last_feedback: str = "Waiting for pose..."
    last_stability_feedback: str = "Elbow stability: Unknown"

    def reset(self) -> None:
        self.counter = 0
        self.stage = None
        self.prev_elbow_position = None
        self.angle_data.clear()
        self.stability_data.clear()
        self.reps_data.clear()
        self.last_angle = None
        self.last_movement = 0.0
        self.last_feedback = "Waiting for pose..."
        self.last_stability_feedback = "Elbow stability: Unknown"
