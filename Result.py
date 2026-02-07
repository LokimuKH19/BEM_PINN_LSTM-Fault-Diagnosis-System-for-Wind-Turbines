import torch
import numpy as np
import matplotlib.pyplot as plt
from freedom import DF_V1_WF1, DF_OMEGA_WF1, DF_THETA_WF1
from BEM_PINN import load_model, predict_fan_forces
from BEM_LSTM import LoadLSTM, denormalize
from BEM_FaultDiagnosis import inject_omega_fault
import time
import matplotlib

# ==========================
# 全局参数
# ==========================
SEQ_LEN = 50
DURATION = 100
EPS = 1e-6

W_F = 0.3      # thrust 权重
W_T = 0.7     # torque 权重(more sensitive)
Tau = 0.95    # 误差限经验阈值


# ==========================
# 模型加载
# ==========================
def load_models(device):
    pinn = load_model(
        "./Model/inflow_angle_model_FullConstraints.pth",
        device=device
    )
    pinn.eval()

    lstm = LoadLSTM().to(device)
    lstm.load_state_dict(
        torch.load("./Model/load_lstm_50s.pth", map_location=device)
    )
    lstm.eval()

    stats = np.load("./Model/lstm_norm_stats.npy", allow_pickle=True).item()
    return pinn, lstm, stats


def temporal_smoothing(scores, window_size=5):
    """对异常分数进行时间窗口平滑"""
    smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    return np.concatenate([scores[:window_size-1], smoothed])


# ==========================
# 单台风机异常评分
# ==========================
def anomaly_score_single_turbine(
        pinn_model,
        lstm_model,
        stats,
        turbine_id,
        device,
        mode="oscillation",
        **kwargs
):
    t0 = 1
    t1 = t0 + DURATION

    V = DF_V1_WF1[t0:t1, turbine_id]
    Theta = DF_THETA_WF1[t0:t1, turbine_id]
    Omega = DF_OMEGA_WF1[t0:t1, turbine_id]

    # --- 人工故障 ---
    Omega_fault = Omega.copy()
    Omega_fault = inject_omega_fault(Omega_fault, mode=mode, **kwargs)

    # --- PINN预测 ---
    Ft_pinn, T_pinn, _, _ = predict_fan_forces(
        pinn_model, V, Omega_fault, Theta, batch_mode=True
    )

    # --- LSTM预测 ---
    Ft_lstm = np.zeros(DURATION)
    T_lstm = np.zeros(DURATION)
    Ft_lstm[:SEQ_LEN] = Ft_pinn[:SEQ_LEN]
    T_lstm[:SEQ_LEN] = T_pinn[:SEQ_LEN]

    for t in range(SEQ_LEN, DURATION - 1):
        Ft_seq = (Ft_pinn[t - SEQ_LEN:t] - stats["Ft_mean"][turbine_id - 1]) / stats["Ft_std"][turbine_id - 1]
        T_seq = (T_pinn[t - SEQ_LEN:t] - stats["T_mean"][turbine_id - 1]) / stats["T_std"][turbine_id - 1]

        seq = torch.tensor(
            np.stack([Ft_seq, T_seq], axis=1),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = lstm_model(seq).cpu().numpy()[0]

        Ft_hat, T_hat = denormalize(pred, stats, turbine_id - 1)
        Ft_lstm[t + 1] = Ft_hat
        T_lstm[t + 1] = T_hat

    # --- 异常检测 ---
    scores = []
    residuals_F = []
    residuals_T = []

    # 计算每个时间点的残差
    for t in range(SEQ_LEN + 1, DURATION):
        err_F = abs(Ft_lstm[t] - Ft_pinn[t - 1]) / stats["Ft_std"][turbine_id - 1]
        err_T = abs(T_lstm[t] - T_pinn[t - 1]) / stats["T_std"][turbine_id - 1]

        score = W_F * err_F + W_T * err_T
        scores.append(score)
        residuals_F.append(err_F)
        residuals_T.append(err_T)

    scores = np.array(scores)
    # Sigmoid映射(tau为经验阈值)
    final_score = max(0, 2 * torch.sigmoid(torch.tensor(np.mean(scores)-Tau)).numpy() - 1)
    return final_score


def plot_fault_probability_heatmap(grid_size=10):
    matplotlib.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pinn, lstm, stats = load_models(device)

    num_turbines = DF_V1_WF1.shape[1] - 1
    probs = np.zeros(num_turbines)

    print("Computing fault probabilities for all turbines...")
    ct1 = time.time()
    sum_score = 0
    for i in range(1, num_turbines+1):
        if i % 2 == 1:
            probs[i-1] = anomaly_score_single_turbine(
                pinn, lstm, stats, i, device, mode="oscillation", A=0.5,
            )
            sum_score += probs[i-1]
        else:
            probs[i-1] = anomaly_score_single_turbine(
                pinn, lstm, stats, i, device, mode="none"
            )
            sum_score += (1 - probs[i-1])
        print(f"Turbine {i:03d}: P_fault = {probs[i-1]:.3f}")
    ct2 = time.time()
    print(f"General Accuracy: {sum_score/num_turbines:.3f}")
    print(f"Time Consumed: {ct2-ct1:.3f}s")

    # reshape 成 grid
    if grid_size**2 != num_turbines:
        raise ValueError("Grid size does not match number of turbines")
    probs_grid = probs.reshape((grid_size, grid_size))

    # 绘图
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(
        probs_grid,
        cmap="cividis",
        vmin=0,
        vmax=1,
        origin="lower"
    )

    # 横轴：个位数 0~9
    ax.set_xticks(np.arange(grid_size))
    ax.set_xticklabels([str(i) for i in range(grid_size)], fontsize=20)

    # 纵轴：十位数 0~9（对应风机编号十位）
    ax.set_yticks(np.arange(grid_size))
    ax.set_yticklabels([str(i) for i in range(grid_size)], fontsize=20)

    ax.set_xlabel("WT Index (ones digit)", fontsize=20)
    ax.set_ylabel("WT Index (tens digit)", fontsize=20)
    ax.set_title("Fault Probability Results", fontsize=22)

    # 每个格子标注概率
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{probs_grid[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if probs_grid[i,j] > 0.5 else "white",
                    fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fault probability", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()


# ==========================
# main
# ==========================
if __name__ == "__main__":
    plot_fault_probability_heatmap()
