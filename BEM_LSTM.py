# BEM_LSTM.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ===== 1. 导入已有模块 =====
from freedom import (
    DF_V1_WF1, DF_OMEGA_WF1, DF_THETA_WF1
)

from BEM_PINN import (
    InflowAngleNet,
    load_model,
    predict_fan_forces,
    set_seed
)





# ==========================
# 2. LSTM 网络定义
# ==========================
class LoadLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)  # Ft, T

    def forward(self, x):
        # x: [B, seq_len, 2]
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # 最后一个时间步
        out = self.fc(out)
        return out            # [B, 2]


# ==========================
# 3. 用 PINN 生成 LSTM 训练数据
# ==========================

# 用于归一化的函数
def compute_normalization_stats(Ft_all, T_all):
    """
    Ft_all, T_all: [time, turbines]
    """
    stats = {}
    stats["Ft_mean"] = np.mean(Ft_all, axis=0)
    stats["Ft_std"]  = np.std(Ft_all, axis=0) + 1e-8
    stats["T_mean"]  = np.mean(T_all, axis=0)
    stats["T_std"]   = np.std(T_all, axis=0) + 1e-8
    return stats


def generate_lstm_dataset(pinn_model, seq_len=50):
    """
    使用 PINN 生成载荷时间序列
    返回:
    X: [samples, seq_len, 2]
    y: [samples, 2]
    """

    Ft_all = []
    T_all = []

    print("Generating PINN-based load time series...")

    for t in range(DF_V1_WF1.shape[0]):
        V = DF_V1_WF1[t, 1:]
        Omega = DF_OMEGA_WF1[t, 1:]
        Theta = DF_THETA_WF1[t, 1:]

        Ft, T, _, _ = predict_fan_forces(
            pinn_model,
            V, Omega, Theta,
            batch_mode=True
        )

        Ft_all.append(Ft)
        T_all.append(T)

    Ft_all = np.array(Ft_all)  # [time, turbines]
    T_all = np.array(T_all)

    # ===== 构造 LSTM 数据 =====
    X_list, y_list = [], []
    stats = compute_normalization_stats(Ft_all, T_all)

    for i in range(NUM_TURBINES):
        for t in range(seq_len, Ft_all.shape[0] - 1):
            Ft_seq = (Ft_all[t - seq_len:t, i] - stats["Ft_mean"][i]) / stats["Ft_std"][i]
            T_seq = (T_all[t - seq_len:t, i] - stats["T_mean"][i]) / stats["T_std"][i]

            seq = np.stack([Ft_seq, T_seq], axis=1)

            target = np.array([
                (Ft_all[t + 1, i] - stats["Ft_mean"][i]) / stats["Ft_std"][i],
                (T_all[t + 1, i] - stats["T_mean"][i]) / stats["T_std"][i]
            ])

            X_list.append(seq)
            y_list.append(target)

    X = torch.tensor(np.array(X_list), dtype=torch.float32, device=device)
    y = torch.tensor(np.array(y_list), dtype=torch.float32, device=device)
    # 存储归一化数据
    np.save("./model/lstm_norm_stats.npy", stats)

    print(f"LSTM dataset generated: X={X.shape}, y={y.shape}")
    return X, y


# ==========================
# 4. LSTM 训练函数
# ==========================
def train_lstm(model, X, y, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(1, epochs + 1):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[LSTM] Epoch {epoch:4d} | Loss = {loss.item():.3e}")


# 反归一化器
def denormalize(pred, stats, i):
    Ft = pred[0] * stats["Ft_std"][i] + stats["Ft_mean"][i]
    T  = pred[1] * stats["T_std"][i]  + stats["T_mean"][i]
    return Ft, T


# 观察器
import matplotlib.pyplot as plt


# 用于观察
def monitor_single_turbine(
    pinn_model,
    lstm_model,
    stats,
    turbine_id,
    start_t=0,
    duration=100,
    seq_len=50
):
    """
    对单台风机进行在线监控，对比 PINN vs LSTM
    """

    Ft_pinn, T_pinn = [], []
    Ft_lstm, T_lstm = [], []

    # ===== 1. 先生成 PINN 真值 =====
    for t in range(start_t, start_t + duration):
        V = DF_V1_WF1[t, turbine_id]
        Omega = DF_OMEGA_WF1[t, turbine_id]
        Theta = DF_THETA_WF1[t, turbine_id]

        Ft, T, _, _ = predict_fan_forces(
            pinn_model, V, Omega, Theta
        )

        Ft_pinn.append(Ft)
        T_pinn.append(T)

    Ft_pinn = np.array(Ft_pinn)
    T_pinn = np.array(T_pinn)

    # ===== 2. LSTM 递推预测 =====
    for t in range(seq_len, duration - 1):
        Ft_seq = (Ft_pinn[t-seq_len:t] - stats["Ft_mean"][turbine_id-1]) \
                 / stats["Ft_std"][turbine_id-1]
        T_seq = (T_pinn[t-seq_len:t] - stats["T_mean"][turbine_id-1]) \
                 / stats["T_std"][turbine_id-1]

        seq = np.stack([Ft_seq, T_seq], axis=1)
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = lstm_model(seq).cpu().numpy()[0]

        Ft_hat, T_hat = denormalize(
            pred, stats, turbine_id-1
        )

        Ft_lstm.append(Ft_hat)
        T_lstm.append(T_hat)

    # 对齐时间轴
    t_axis_pinn = np.arange(start_t + seq_len, start_t + duration - 1)
    t_axis_lstm = t_axis_pinn + 1

    # ===== 3. 绘图 =====
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(t_axis_pinn, Ft_pinn[seq_len:-1], label="PINN")
    ax[0].plot(t_axis_lstm, Ft_lstm, "--", label="LSTM")
    ax[0].set_ylabel("Thrust [N]")
    ax[0].legend()

    ax[1].plot(t_axis_pinn, T_pinn[seq_len:-1], label="PINN")
    ax[1].plot(t_axis_lstm, T_lstm, "--", label="LSTM")
    ax[1].set_ylabel("Torque [Nm]")
    ax[1].set_xlabel("Time [s]")
    ax[1].legend()

    plt.suptitle(f"Turbine {turbine_id} | Online Load Monitoring")
    plt.tight_layout()
    plt.show()


# 读取模型的调用
def load_pinn_and_lstm(
    pinn_path="./model/inflow_angle_model_FullConstraints.pth",
    lstm_path="./model/load_lstm.pth",
    device="cpu"
):
    # PINN
    pinn_model = load_model(
        pinn_path,
        device=device
    )
    pinn_model.eval()

    # LSTM
    lstm_model = LoadLSTM().to(device)
    lstm_model.load_state_dict(
        torch.load(lstm_path, map_location=device)
    )
    lstm_model.eval()

    # 归一化参数
    stats = np.load(
        "./model/lstm_norm_stats.npy",
        allow_pickle=True
    ).item()

    print("PINN & LSTM loaded successfully.")
    return pinn_model, lstm_model, stats


# ==========================
# 5. 主程序
# ==========================
if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEQ_LEN = 50  # 使用前50s训练
    NUM_FEATURES = 2  # Ft, T
    NUM_TURBINES = DF_V1_WF1.shape[1] - 1
    # ===== 加载 PINN =====
    print("Loading PINN model...")
    pinn_model = load_model(
        "./model/inflow_angle_model_FullConstraints.pth",
        device=device
    )

    # ===== 生成 LSTM 数据 =====
    X_lstm, y_lstm = generate_lstm_dataset(
        pinn_model,
        seq_len=SEQ_LEN
    )

    # ===== 初始化 & 训练 LSTM =====
    lstm_model = LoadLSTM().to(device)

    print("Training LSTM...")
    train_lstm(lstm_model, X_lstm, y_lstm, epochs=3000)
    torch.save(lstm_model.state_dict(), "./model/load_lstm_50s.pth")
    print("LSTM model saved.")

    # ===== 在线监控示例 =====
    pinn_model, lstm_model, stats = load_pinn_and_lstm(
        pinn_path="./model/inflow_angle_model_FullConstraints.pth",
        lstm_path="./model/load_lstm_50s.pth",
        device=device
    )
    monitor_single_turbine(
        pinn_model=pinn_model,
        lstm_model=lstm_model,
        stats=stats,
        turbine_id=6,   # 查询风机
        start_t=1,
        duration=99
    )
