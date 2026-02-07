# BEM_FaultDetection.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from freedom import (
    DF_V1_WF1,
    DF_OMEGA_WF1,
    DF_THETA_WF1
)

from BEM_PINN import (
    load_model,
    predict_fan_forces
)

from BEM_LSTM import (
    LoadLSTM,
    denormalize
)

SEQ_LEN = 50
EPS = 1e-6


# ==========================
# 模型加载
# ==========================
def load_models(device, pinn_path, lstm_path):
    pinn_model = load_model(pinn_path, device=device)
    pinn_model.eval()

    lstm_model = LoadLSTM().to(device)
    lstm_model.load_state_dict(
        torch.load(lstm_path, map_location=device)
    )
    lstm_model.eval()

    stats = np.load("./model/lstm_norm_stats.npy", allow_pickle=True).item()
    print("PINN & LSTM models loaded.")
    return pinn_model, lstm_model, stats


# ==========================
# 故障注入
# ==========================
def inject_omega_fault(Omega, mode="drift", **kwargs):
    Omega_fault = Omega.copy()

    if mode == "drift":
        eps = kwargs.get("eps", 0.2)
        for i in range(len(Omega)):
            Omega_fault[i] *= (1 + eps * i)

    elif mode == "bias":
        delta = kwargs.get("delta", 0.2)
        Omega_fault += delta

    elif mode == "oscillation":

        A = kwargs.get("A", 0.5)
        f = kwargs.get("f", 0.1)
        t = np.arange(len(Omega))
        Omega_fault *= (1 + A * np.sin(2 * np.pi * f * t))

    elif mode == "none":
        return Omega_fault

    return Omega_fault


# ==========================
# 故障检测主函数
# ==========================
def fault_detection_single_turbine(
    pinn_model,
    lstm_model,
    stats,
    turbine_id,
    duration=100,
    device="cuda",
    mode="drift",    # 传入故障类型
):

    t0 = 1
    t1 = t0 + duration
    time_axis = np.arange(t0, t1)

    # ===== 原始数据 =====
    V = DF_V1_WF1[t0:t1, turbine_id]
    Theta = DF_THETA_WF1[t0:t1, turbine_id]
    Omega = DF_OMEGA_WF1[t0:t1, turbine_id]

    # ===== 故障 Omega =====
    Omega_fault = Omega.copy()
    Omega_fault[SEQ_LEN:] = inject_omega_fault(Omega[SEQ_LEN:], mode=mode)

    # ===== PINN（长度 duration-1）=====
    Ft_pinn_norm, T_pinn_norm, _, _ = predict_fan_forces(
        pinn_model, V, Omega, Theta, batch_mode=True
    )
    Ft_pinn_fault, T_pinn_fault, _, _ = predict_fan_forces(
        pinn_model, V, Omega_fault, Theta, batch_mode=True
    )

    # ===== LSTM（长度 duration）=====
    Ft_lstm = np.zeros(duration)
    T_lstm = np.zeros(duration)

    Ft_lstm[:SEQ_LEN] = Ft_pinn_norm[:SEQ_LEN]
    T_lstm[:SEQ_LEN] = T_pinn_norm[:SEQ_LEN]

    for t in range(SEQ_LEN, duration - 1):
        Ft_seq = (Ft_pinn_norm[t-SEQ_LEN:t] - stats["Ft_mean"][turbine_id-1]) \
                 / stats["Ft_std"][turbine_id-1]
        T_seq = (T_pinn_norm[t-SEQ_LEN:t] - stats["T_mean"][turbine_id-1]) \
                 / stats["T_std"][turbine_id-1]

        seq = torch.tensor(
            np.stack([Ft_seq, T_seq], axis=1),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = lstm_model(seq).cpu().numpy()[0]

        Ft_hat, T_hat = denormalize(pred, stats, turbine_id-1)

        Ft_lstm[t + 1] = Ft_hat
        T_lstm[t + 1] = T_hat

    # ===== 相对误差（对齐 t）=====
    rel_F_norm = np.zeros(duration)
    rel_T_norm = np.zeros(duration)
    rel_F_fault = np.zeros(duration)
    rel_T_fault = np.zeros(duration)

    for t in range(SEQ_LEN + 1, duration):
        rel_F_norm[t] = abs(Ft_lstm[t] - Ft_pinn_norm[t-1]) / (abs(Ft_pinn_norm[t-1]) + EPS)
        rel_T_norm[t] = abs(T_lstm[t] - T_pinn_norm[t-1]) / (abs(T_pinn_norm[t-1]) + EPS)

        rel_F_fault[t] = abs(Ft_lstm[t] - Ft_pinn_fault[t-1]) / (abs(Ft_pinn_fault[t-1]) + EPS)
        rel_T_fault[t] = abs(T_lstm[t] - T_pinn_fault[t-1]) / (abs(T_pinn_fault[t-1]) + EPS)

    # ==========================
    # 绘图
    # ==========================
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # --- Thrust ---
    ax[0].plot(time_axis[1:], Ft_pinn_fault, label="PINN-From Sensors")
    ax[0].plot(time_axis+1, Ft_lstm, "--", label="LSTM-Predicted from History")
    ax[0].axvline(t0 + SEQ_LEN, color="k", linestyle=":")
    ax[0].set_ylabel("Thrust")
    ax[0].legend()

    # --- Torque ---
    ax[1].plot(time_axis[1:], T_pinn_fault, label="PINN-From Sensors")
    ax[1].plot(time_axis+1, T_lstm, "--", label="LSTM-Predicted from History")
    ax[1].axvline(t0 + SEQ_LEN, color="k", linestyle=":")
    ax[1].set_ylabel("Torque")
    ax[1].legend()

    # --- Relative Error ---
    ax[2].plot(time_axis, rel_F_norm, label="Rel. error F (normal)")
    ax[2].plot(time_axis, rel_T_norm, label="Rel. error T (normal)")
    ax[2].plot(time_axis, rel_F_fault, "--", label="Rel. error F (with fault)")
    ax[2].plot(time_axis, rel_T_fault, "--", label="Rel. error T (with fault)")

    ax[2].axvline(t0 + SEQ_LEN, color="k", linestyle=":")
    ax[2].set_ylabel("Relative error")
    ax[2].set_xlabel("Time [s]")
    ax[2].legend()

    plt.suptitle(f"Turbine {turbine_id} | PINN–LSTM Fault Detection")
    plt.tight_layout()
    plt.show()


# ==========================
# main
# ==========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pinn_model, lstm_model, stats = load_models(
        device,
        "./Model/inflow_angle_model_FullConstraints.pth",
        "./Model/load_lstm_50s.pth"
    )

    fault_detection_single_turbine(
        pinn_model,
        lstm_model,
        stats,
        turbine_id=42,
        duration=100,
        device=device,
        mode='oscillation'    # drift/bias/oscillation/none，系统响应故障的模式
    )
