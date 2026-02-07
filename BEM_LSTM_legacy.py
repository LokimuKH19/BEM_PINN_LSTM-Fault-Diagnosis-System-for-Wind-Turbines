# BEM_LSTM.py
# 事实上这种LSTM最好做成分类器的形式，直接判定载荷序列对应哪种运行状态，但目前我们没有收集到足够的标签数据
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
set_seed(1024)


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
def train_lstm(model, X, y, epochs=200, lr=1e-3, split=0.8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    perm = torch.randperm(X.size(0))
    X = X[perm]
    y = y[perm]

    model.train()
    # 数据类型：[每台风机49段*100]共4900条
    single_num_segment = int(X.shape[0] / NUM_TURBINES)
    split_train = int(single_num_segment * split)   # 对每个风机的数据取这么多条

    X_train = torch.vstack([X[i:i + single_num_segment][:split_train] for i in range(0, len(X), single_num_segment)])
    y_train = torch.vstack([y[i:i + single_num_segment][:split_train] for i in range(0, len(y), single_num_segment)])

    X_test = torch.vstack([X[i:i + single_num_segment][split_train:] for i in range(0, len(X), single_num_segment)])
    y_test = torch.vstack([y[i:i + single_num_segment][split_train:] for i in range(0, len(y), single_num_segment)])

    for epoch in range(1, epochs + 1):
        pred_train = model(X_train)
        loss_train = loss_fn(pred_train, y_train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                pred_test = model(X_test)
                loss_test = loss_fn(pred_test, y_test)
            print(f"[LSTM] Epoch {epoch:4d} | Train Loss = {loss_train.item():.3e} | Test Loss = {loss_test.item():.3e}")


# 反归一化器
def denormalize(pred, stats, i):
    Ft = pred[0] * stats["Ft_std"][i] + stats["Ft_mean"][i]
    T  = pred[1] * stats["T_std"][i]  + stats["T_mean"][i]
    return Ft, T


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


def evaluate_lstm_error_all_turbines(
    pinn_model,
    lstm_model,
    stats,
    seq_len=50,
    eps=1e-8,
    device="cpu"
):
    NUM_TURBINES = DF_V1_WF1.shape[1] - 1
    T_TOTAL = DF_V1_WF1.shape[0]

    train_err_F, train_err_T = [], []
    test_err_F, test_err_T = [], []

    for tid in range(1, NUM_TURBINES + 1):

        # ===== PINN 真值 =====
        Ft_pinn, T_pinn = [], []
        for t in range(T_TOTAL):
            V = DF_V1_WF1[t, tid]
            Omega = DF_OMEGA_WF1[t, tid]
            Theta = DF_THETA_WF1[t, tid]
            Ft, T, _, _ = predict_fan_forces(pinn_model, V, Omega, Theta)
            Ft_pinn.append(Ft)
            T_pinn.append(T)

        Ft_pinn = np.array(Ft_pinn)
        T_pinn = np.array(T_pinn)

        # ===== LSTM 递推 =====
        Ft_lstm = np.zeros(T_TOTAL)
        T_lstm = np.zeros(T_TOTAL)

        Ft_lstm[:seq_len] = Ft_pinn[:seq_len]
        T_lstm[:seq_len] = T_pinn[:seq_len]

        for t in range(seq_len, T_TOTAL - 1):
            Ft_seq = (Ft_pinn[t-seq_len:t] - stats["Ft_mean"][tid-1]) / stats["Ft_std"][tid-1]
            T_seq  = (T_pinn[t-seq_len:t]  - stats["T_mean"][tid-1])  / stats["T_std"][tid-1]

            seq = torch.tensor(
                np.stack([Ft_seq, T_seq], axis=1),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = lstm_model(seq).cpu().numpy()[0]

            Ft_hat = pred[0] * stats["Ft_std"][tid-1] + stats["Ft_mean"][tid-1]
            T_hat  = pred[1] * stats["T_std"][tid-1]  + stats["T_mean"][tid-1]

            Ft_lstm[t+1] = Ft_hat
            T_lstm[t+1]  = T_hat

        # ===== 相对误差（时间对齐）=====
        for t in range(seq_len, T_TOTAL - 1):
            err_F = abs(Ft_lstm[t+1] - Ft_pinn[t]) / (abs(Ft_pinn[t]) + eps)
            err_T = abs(T_lstm[t+1] - T_pinn[t])  / (abs(T_pinn[t])  + eps)

            if t < 90:
                train_err_F.append(err_F)
                train_err_T.append(err_T)
            else:
                test_err_F.append(err_F)
                test_err_T.append(err_T)

    return {
        "train_F_mean": np.mean(train_err_F),
        "train_F_max":  np.max(train_err_F),
        "train_T_mean": np.mean(train_err_T),
        "train_T_max":  np.max(train_err_T),
        "test_F_mean":  np.mean(test_err_F),
        "test_F_max":   np.max(test_err_F),
        "test_T_mean":  np.mean(test_err_T),
        "test_T_max":   np.max(test_err_T),
    }


# ==========================
# 5. 主程序
# ==========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEQ_LEN = 50  # 基于50s序列进行前向预测
    SPLIT = 0.9   # 用前90%数据训练，剩余留做验证
    NUM_FEATURES = 2  # Ft, T
    NUM_TURBINES = DF_V1_WF1.shape[1] - 1
    print(NUM_TURBINES)
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
    # train_lstm(lstm_model, X_lstm, y_lstm, epochs=2000)
    # torch.save(lstm_model.state_dict(), "./model/load_lstm_50s_test.pth")
    print("LSTM model saved.")

    # ===== 在线监控示例 =====
    pinn_model, lstm_model, stats = load_pinn_and_lstm(
        pinn_path="./Model/inflow_angle_model_FullConstraints.pth",
        lstm_path="./Model/load_lstm_50s_test.pth",
        device=device
    )
    monitor_single_turbine(
        pinn_model=pinn_model,
        lstm_model=lstm_model,
        stats=stats,
        turbine_id=90,   # 查询风机
        start_t=1,
        duration=99
    )

    print("\n===== Evaluating LSTM relative errors on all turbines =====")

    err_stats = evaluate_lstm_error_all_turbines(
        pinn_model=pinn_model,
        lstm_model=lstm_model,
        stats=stats,
        seq_len=SEQ_LEN,
        device=device
    )

    print("Training period (first 90s):")
    print(f"  Ft mean = {err_stats['train_F_mean']:.3e}, max = {err_stats['train_F_max']:.3e}")
    print(f"  T  mean = {err_stats['train_T_mean']:.3e}, max = {err_stats['train_T_max']:.3e}")

    print("Testing period (last 10s):")
    print(f"  Ft mean = {err_stats['test_F_mean']:.3e}, max = {err_stats['test_F_max']:.3e}")
    print(f"  T  mean = {err_stats['test_T_mean']:.3e}, max = {err_stats['test_T_max']:.3e}")

