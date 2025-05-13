import pandas as pd
import matplotlib.pyplot as plt

# 直接画出loss
def draw_loss(csv_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["Step"], df["Value"], label="Training Loss", color="blue")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def smooth_curve(values, window_size):
    return values.rolling(window=window_size, min_periods=1, center=True).mean()

def exponential_moving_average(values, alpha=0.1):
    smoothed = [values[0]]  # 初始化第一个值
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

def plot_loss_curve(csv_file, window_size=10, alpha=0.1, smoothing_type='simple', save_path=None):
    df = pd.read_csv(csv_file)

    if smoothing_type == 'simple':
        smoothed_values = smooth_curve(df["Value"], window_size)
        smoothing_label = f"Smoothed Loss (Window size={window_size})"
    elif smoothing_type == 'ema':
        smoothed_values = exponential_moving_average(df["Value"], alpha)
        smoothing_label = f"EMA Smoothed Loss (alpha={alpha})"
    else:
        raise ValueError("Invalid smoothing type. Choose 'simple' or 'ema'.")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Step"], df["Value"], label="Original Loss", color="blue", alpha=0.3)  # 原始曲线
    plt.plot(df["Step"], smoothed_values, label=smoothing_label, color="red", linewidth=2)  # 平滑后的曲线

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve with Smoothing")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")

    plt.show()

