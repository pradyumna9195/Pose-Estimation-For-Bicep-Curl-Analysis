import matplotlib.pyplot as plt


MIN_ANGLE = 30
MAX_ANGLE = 160


def plot_angle_over_time(angle_data: list[float]):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(angle_data, label="Elbow Angle", color="blue")
    ax.axhline(y=MIN_ANGLE, color="red", linestyle="--", label="Min Angle (30°)")
    ax.axhline(y=MAX_ANGLE, color="green", linestyle="--", label="Max Angle (160°)")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Elbow Angle (degrees)")
    ax.set_title("Elbow Angle Over Time")
    ax.legend()
    return fig


def plot_stability_hist(stability_data: list[float]):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(stability_data, bins=20, color="blue", alpha=0.7)
    ax.set_xlabel("Elbow Movement")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Elbow Movement Stability")
    return fig


def plot_angle_vs_stability(angle_data: list[float], stability_data: list[float]):
    fig, ax = plt.subplots(figsize=(10, 4))
    min_len = min(len(angle_data), len(stability_data))
    ax.scatter(angle_data[:min_len], stability_data[:min_len])
    ax.set_xlabel("Elbow Angle (degrees)")
    ax.set_ylabel("Elbow Movement")
    ax.set_title("Scatter Plot of Angle vs. Stability")
    return fig
