import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from freedom import (
    R_, DR_, C_, AT_, Const,
    calc_ab, calc_cl_cd, calc_sigma,
    DF_V1_WF1, DF_OMEGA_WF1, DF_THETA_WF1
)
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Operator NN
# ==========================
class InflowAngleNet(nn.Module):
    def __init__(self, Nr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, Nr)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [3] -> [1, 3]
        raw = self.net(x)  # [B, Nr]
        V = x[:, 0:1]
        Omega = x[:, 1:2]
        r = torch.tensor(R_, device=x.device).view(1, -1)
        # 确保和流向确定的角度具有相同的符号
        phi0 = torch.atan2(V, Omega * r)  # [B, Nr]
        # 网络本质上只预测实际的入流角和V/Omega*r确定的角度相差了多少
        phi = phi0 * (1.0 + 0.3 * torch.tanh(raw))
        return phi


# ==========================
# Ft/T batch 计算
# ==========================
def calc_ftp_from_phi_batch(phi, V, Omega, Theta):
    """
    phi: [B,Nr] numpy 或 Tensor, 网络预测的 φ
    V/Omega/Theta: [B]
    返回 Ft/T Tensor
    """
    phi_tensor = torch.tensor(phi, dtype=torch.float32, device=device) if not isinstance(phi, torch.Tensor) else phi
    V_tensor = torch.tensor(V, dtype=torch.float32, device=device) if not isinstance(V, torch.Tensor) else V
    Omega_tensor = torch.tensor(Omega, dtype=torch.float32, device=device) if not isinstance(Omega,
                                                                                             torch.Tensor) else Omega
    Theta_tensor = torch.tensor(Theta, dtype=torch.float32, device=device) if not isinstance(Theta,
                                                                                             torch.Tensor) else Theta

    # 向量化 Ft/T 计算
    B, Nr = phi_tensor.shape
    r_t = torch.tensor(R_, dtype=torch.float32, device=device).view(1, Nr)
    dr_t = torch.tensor(DR_, dtype=torch.float32, device=device).view(1, Nr)
    C_t = torch.tensor(C_, dtype=torch.float32, device=device).view(1, Nr)
    AT_t = torch.tensor(AT_, dtype=torch.float32, device=device).view(1, Nr)

    V_t = V_tensor.view(B, 1)
    Omega_t = Omega_tensor.view(B, 1)
    Theta_t = Theta_tensor.view(B, 1)

    W = torch.sqrt((V_t * torch.sin(phi_tensor)) ** 2 + (Omega_t * r_t * torch.cos(phi_tensor)) ** 2)
    alpha = phi_tensor - (Theta_t + AT_t) / 180 * np.pi

    # 修复: 转置 alpha 使其符合 calc_cl_cd 的输入格式 (Nr, B)
    alpha_np = alpha.detach().cpu().numpy()  # [B, Nr]

    # 无论B是多少，都转置为 [Nr, B]
    alpha_np = alpha_np.T  # [Nr, B]

    cl_np, cd_np = calc_cl_cd(alpha_np)  # 用 numpy 插值，返回形状为 [Nr, B]

    # 将 cl, cd 转回原来的形状 [B, Nr]
    cl = torch.tensor(cl_np, dtype=torch.float32, device=device).T  # [B, Nr]
    cd = torch.tensor(cd_np, dtype=torch.float32, device=device).T  # [B, Nr]

    cn = cl * torch.cos(phi_tensor) + cd * torch.sin(phi_tensor)
    ct = cl * torch.sin(phi_tensor) - cd * torch.cos(phi_tensor)
    # 用BEM迭代式反算a,b
    sigma = calc_sigma(R_, device=phi_tensor.device).view(1, Nr)
    a = 1 / (4 * torch.sin(phi_tensor) ** 2 / sigma / cn + 1)
    b = 1 / (4 * torch.cos(phi_tensor) * torch.sin(phi_tensor) / sigma / ct - 1)
    # 计算叶片的方程损失分布
    Lft = (Const.B * W ** 2 * C_t * cn) / (8 * torch.pi * r_t * V_t ** 2) - a * (1 - a)  # 轴向力平衡
    Lts = (Const.B * W ** 2 * C_t * ct) / (8 * torch.pi * r_t ** 2 * Omega_t * V_t) - b * (1 - a)  # 切向力矩平衡
    dFt = 0.5 * Const.rho * Const.B * W ** 2 * C_t * cn * dr_t
    dT = 0.5 * Const.rho * Const.B * W ** 2 * C_t * ct * r_t * dr_t

    # 计算实时的Ft和T
    Ft = torch.sum(dFt, dim=1)
    T = torch.sum(dT, dim=1)
    return Ft, T, Lft, Lts


# ==========================
# 数据集准备 (numpy -> Tensor)
# ==========================
TRAIN_ROWS = 10
TEST_ROWS = 90
NUM_TURBINES = DF_V1_WF1.shape[1] - 1


def prepare_dataset(train=True):
    if train:
        t_rows = range(0, TRAIN_ROWS)
    else:
        t_rows = range(DF_V1_WF1.shape[0] - TEST_ROWS, DF_V1_WF1.shape[0])

    X_list, Y_phi_list, meta_list = [], [], []

    for t in t_rows:
        V_row = DF_V1_WF1[t, 1:]        # shape=(NUM_TURBINES,)
        Omega_row = DF_OMEGA_WF1[t, 1:]
        Theta_row = DF_THETA_WF1[t, 1:]

        # 计算 BEM φ
        a, b = calc_ab(R_, V_row, Omega_row, Theta_row, eps=1e-5)

        # 将 R_ 扩展到与 a,b 形状一致
        r_tile = np.tile(R_[:, np.newaxis], (1, V_row.shape[0]))  # (17,NUM_TURBINES)
        phi = np.arctan2((1 - a) * V_row[np.newaxis, :], (1 + b) * Omega_row[np.newaxis, :] * r_tile)  # (17, NUM_TURBINES)

        # 保存每台涡轮的数据
        for i in range(NUM_TURBINES):
            X_list.append([V_row[i], Omega_row[i], Theta_row[i]])
            Y_phi_list.append(phi[:, i])  # i 对应列索引
            meta_list.append((t, i+1))   # 风机编号和数据列对应

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32, device=device)
    Y_phi_tensor = torch.tensor(np.array(Y_phi_list), dtype=torch.float32, device=device)
    Ft_tensor, T_tensor, _, _ = calc_ftp_from_phi_batch(Y_phi_tensor, X_tensor[:, 0], X_tensor[:, 1], X_tensor[:, 2])
    return X_tensor, Y_phi_tensor, Ft_tensor, T_tensor, meta_list


# 光滑性约束
def smoothness_loss(phi, dr):
    """
    phi: [B, Nr] tensor
    dr:  [1, Nr] or scalar, 节点间距
    返回: 平均光滑损失
    """
    # 二阶差分 ∆²φ / ∆r²
    d2phi = phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]  # shape [B, Nr-2]
    loss = torch.mean((d2phi / (dr[:,1:-1]**2))**2)
    return loss


# 倾向于下凸约束
def concave_down_loss(phi):
    """
    phi: [B, Nr] tensor
    返回: 平均下凸约束损失
    """
    # 二阶差分
    d2phi = phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]  # shape [B, Nr-2]
    # 下凸要求 phi'' <= 0, 违反时惩罚
    loss = torch.mean(torch.relu(d2phi))
    return loss


# ==========================
# φ 可视化
# ==========================
def plot_phi(phi_bem, phi_nn, t, i, epoch):
    plt.figure(figsize=(6,4))
    plt.plot(R_, phi_bem, label="BEM φ")
    plt.plot(R_, phi_nn, "--", label="NN φ")
    plt.xlabel("r")
    plt.ylabel("φ [rad]")
    plt.title(f"φ(r) | epoch={epoch}, t={t}, turbine={i}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 画图专用的函数
def calc_ftp_single_phi(phi, V, Omega, Theta):
    """
    使用单个 φ 计算 Ft/T，可用于绘图或 compare_all_turbines
    phi: 1D numpy array, shape=(Nr,)
    V/Omega/Theta: float
    返回 Ft, T: float
    """
    phi = np.array(phi).flatten()  # 保证 shape=(Nr,)
    r = np.array(R_)
    dr = np.array(DR_)
    C = np.array(C_)
    AT = np.array(AT_)

    # 空气动力学升阻力
    alpha = phi - (Theta + AT)/180*np.pi
    cl, cd = calc_cl_cd(alpha)
    cn = cl*np.cos(phi) + cd*np.sin(phi)
    ct = cl*np.sin(phi) - cd*np.cos(phi)

    W = np.sqrt((V*np.sin(phi))**2 + (Omega*r*np.cos(phi))**2)

    dFt = 0.5*Const.rho*Const.B*W**2*C*cn*dr
    dT = 0.5*Const.rho*Const.B*W**2*C*ct*r*dr

    Ft = np.sum(dFt)
    T = np.sum(dT)
    return Ft, T


# ==========================
# Ft/T 可视化
# ==========================
def compare_all_turbines(model, t):
    Ft_bem, Ft_nn, T_bem, T_nn = [], [], [], []

    for i in range(NUM_TURBINES):
        V1 = DF_V1_WF1[t, i+1]
        Omega = DF_OMEGA_WF1[t, i+1]
        Theta = DF_THETA_WF1[t, i+1]

        # BEM φ
        a, b = calc_ab(R_, np.array([V1]), np.array([Omega]), np.array([Theta]), eps=1e-5)
        # 取第 0 列，因为我们只输入一个涡轮
        phi_bem = np.arctan2((1-a)[:,0] * V1, (1+b)[:,0] * Omega * np.array(R_))
        Ft, T = calc_ftp_single_phi(phi_bem, V1, Omega, Theta)
        Ft_bem.append(Ft)
        T_bem.append(T)

        # NN φ
        x = torch.tensor([V1, Omega, Theta], dtype=torch.float32, device=device)
        phi_nn = model(x).detach().cpu().numpy().flatten()
        Ft, T = calc_ftp_single_phi(phi_nn, V1, Omega, Theta)
        Ft_nn.append(Ft)
        T_nn.append(T)

    # 绘图
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(Ft_bem, label="BEM")
    ax[0].plot(Ft_nn, "--", label="NN")
    ax[0].set_title("Tower Thrust")
    ax[0].legend()
    ax[1].plot(T_bem, label="BEM")
    ax[1].plot(T_nn, "--", label="NN")
    ax[1].set_title("Aerodynamic Torque")
    ax[1].legend()
    plt.suptitle(f"t={t}")
    plt.tight_layout()
    plt.show()


# 物理损失: 包含全局弱守恒约束和局部约束
def phys_loss(LFt, LTs, dr):
    """
    LFt, LTs: [B, Nr]
    dr:       [1, Nr]
    """

    # 局部一致性
    loc_ft = torch.sum(LFt**2 * dr, dim=1)
    loc_ts = torch.sum(LTs**2 * dr, dim=1)

    loss_ = torch.mean(loc_ft + loc_ts)
    return loss_


# 便于调试，防止反复拆分X
def compute_phys_from_phi(phi, X):
    """
    phi: [B, Nr]
    X:   [B, 3]  (e.g. V, omega, pitch)
    """
    Ft, T, LFt, LTs = calc_ftp_from_phi_batch(
        phi,
        X[:, 0],
        X[:, 1],
        X[:, 2]
    )
    return Ft, T, LFt, LTs


# 保存整个模型（包括结构和参数）
def save_model(model, path="inflow_angle_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")


# 读取模型（注意要和保存时的结构一致）
def load_model(path="inflow_angle_model.pth", Nr=len(R_), device="cpu"):
    model = InflowAngleNet(Nr).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # 切换到推理模式
    print(f"模型已加载 {path}")
    return model


# ==========================
# predict_fan_forces 函数 (保持原样)
# ==========================
def predict_fan_forces(model, V, Omega, Theta, batch_mode=False):
    """
    使用训练好的模型预测风机的力和力矩

    参数:
    model: 训练好的InflowAngleNet模型
    V: 风速，可以是标量或数组
    Omega: 转速，可以是标量或数组
    Theta: 桨距角，可以是标量或数组
    batch_mode: 是否批量处理，如果为True，则输入可以是数组

    返回:
    Ft: 推力 [N]
    T: 扭矩 [Nm]
    LFt: 轴向力平衡损失分布
    LTs: 切向力矩平衡损失分布
    """
    device = next(model.parameters()).device

    if batch_mode:
        # 批量处理模式
        if not (hasattr(V, '__len__') and hasattr(Omega, '__len__') and hasattr(Theta, '__len__')):
            raise ValueError("在batch_mode=True时，V, Omega, Theta必须为数组")

        # 确保输入是tensor
        if not isinstance(V, torch.Tensor):
            V = torch.tensor(V, dtype=torch.float32, device=device)
        if not isinstance(Omega, torch.Tensor):
            Omega = torch.tensor(Omega, dtype=torch.float32, device=device)
        if not isinstance(Theta, torch.Tensor):
            Theta = torch.tensor(Theta, dtype=torch.float32, device=device)

        # 确保张量是二维的 (batch_size, 1)
        if V.dim() == 1:
            V = V.view(-1, 1)
        if Omega.dim() == 1:
            Omega = Omega.view(-1, 1)
        if Theta.dim() == 1:
            Theta = Theta.view(-1, 1)

        # 检查形状是否一致
        if not (V.shape == Omega.shape == Theta.shape):
            # 如果形状不一致，尝试调整
            max_len = max(V.shape[0], Omega.shape[0], Theta.shape[0])
            if V.shape[0] != max_len and V.shape[0] == 1:
                V = V.expand(max_len, 1)
            if Omega.shape[0] != max_len and Omega.shape[0] == 1:
                Omega = Omega.expand(max_len, 1)
            if Theta.shape[0] != max_len and Theta.shape[0] == 1:
                Theta = Theta.expand(max_len, 1)

        # 创建输入矩阵
        X = torch.cat([V, Omega, Theta], dim=1)

    else:
        # 单点处理模式
        if isinstance(V, (list, tuple, np.ndarray)):
            V = V[0] if len(V) > 0 else V
        if isinstance(Omega, (list, tuple, np.ndarray)):
            Omega = Omega[0] if len(Omega) > 0 else Omega
        if isinstance(Theta, (list, tuple, np.ndarray)):
            Theta = Theta[0] if len(Theta) > 0 else Theta

        # 转换为标量tensor
        V = torch.tensor([float(V)], dtype=torch.float32, device=device)
        Omega = torch.tensor([float(Omega)], dtype=torch.float32, device=device)
        Theta = torch.tensor([float(Theta)], dtype=torch.float32, device=device)

        # 创建输入矩阵
        X = torch.stack([V, Omega, Theta], dim=1)

    # 确保模型在eval模式
    model.eval()

    with torch.no_grad():
        # 预测入流角
        phi_pred = model(X)

        # 计算力和力矩
        Ft, T, LFt, LTs = calc_ftp_from_phi_batch(
            phi_pred,
            X[:, 0],
            X[:, 1],
            X[:, 2]
        )

    # 如果不是批量模式，返回标量值
    if not batch_mode:
        return Ft.item(), T.item(), LFt.cpu().numpy(), LTs.cpu().numpy()
    else:
        return Ft.cpu().numpy(), T.cpu().numpy(), LFt.cpu().numpy(), LTs.cpu().numpy()


# 使用示例
def example_usage(path):
    """使用predict_fan_forces函数的示例"""
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(path, Nr=len(R_), device=device)

    # 示例1: 单个工况预测
    print("单个工况预测:")
    V, Omega, Theta = 13.0, 1.26, 10.0
    Ft, T, LFt, LTs = predict_fan_forces(model, V, Omega, Theta)
    print(f"V={V} m/s, Omega={Omega} rad/s, Theta={Theta} deg")
    print(f"推力 Ft = {Ft:.2f} N")
    print(f"扭矩 T = {T:.2f} Nm")

    # 示例2: 批量预测
    print("\n批量工况预测:")
    V_batch = np.array([8.0, 10.0, 12.0, 14.0])
    Omega_batch = np.array([1.0, 1.1, 1.2, 1.3])
    Theta_batch = np.array([5.0, 8.0, 12.0, 9.0])

    Ft_batch, T_batch, LFt_batch, LTs_batch = predict_fan_forces(
        model, V_batch, Omega_batch, Theta_batch, batch_mode=True
    )

    for i in range(len(V_batch)):
        print(f"\n工况 {i + 1}:")
        print(f"V={V_batch[i]} m/s, Omega={Omega_batch[i]} rad/s, Theta={Theta_batch[i]} deg")
        print(f"推力 Ft = {Ft_batch[i]:.2f} N")
        print(f"扭矩 T = {T_batch[i]:.2f} Nm")

    return model


# ==========================
# main training
# ==========================
# 在main函数中替换原来的预测代码
if __name__ == "__main__":
    SEED = 1024
    set_seed(SEED)
    model = InflowAngleNet(len(R_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1024
    lambda_data = 1
    lambda_phys = 1e-3    # 物理方程权重
    lambda_smooth = 1e-3  # 光滑权重
    lambda_concave = 1e-3  # 下凸权重

    print("准备数据集...")
    X_train, Y_train_phi, Ft_train, T_train, meta_train = prepare_dataset(train=True)
    X_test, Y_test_phi, Ft_test, T_test, meta_test = prepare_dataset(train=False)
    print("数据生成完毕")

    dr = torch.tensor(DR_, device=device).view(1, -1)

    for epoch in range(1, num_epochs + 1):
        # ===== train =====
        phi_pred = model(X_train)
        # 数据部分
        loss_data = torch.mean((phi_pred - Y_train_phi) ** 2)

        # 物理部分
        _, _, LFt, LTs = compute_phys_from_phi(phi_pred, X_train)
        loss_smooth = smoothness_loss(phi_pred, dr)
        loss_concave = concave_down_loss(phi_pred)
        loss_phys = phys_loss(LFt, LTs, dr)

        loss = lambda_data * loss_data + lambda_phys * loss_phys + lambda_smooth * loss_smooth + lambda_concave * loss_concave

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== test =====
        if epoch % 1 == 0:
            with torch.no_grad():
                phi_test = model(X_test)
                loss_test_data = torch.mean((phi_test - Y_test_phi) ** 2)

                _, _, LFt_t, LTs_t = compute_phys_from_phi(phi_test, X_test)
                loss_test_smooth = smoothness_loss(phi_test, dr)
                loss_test_concave = concave_down_loss(phi_test)
                loss_test_phys = phys_loss(LFt_t, LTs_t, dr)

                loss_test = lambda_data * loss_test_data + lambda_phys * loss_test_phys + lambda_smooth * loss_test_smooth + lambda_concave * loss_test_concave

            print(
                f"Epoch {epoch:5d} | "
                f"train={loss.item():.3e} | "
                f"test={loss_test.item():.3e}"
            )

        if epoch % 500 == 0 and epoch > 0:
            # 挑一个训练数据出来看看是怎么个事
            t_train, i_train = meta_train[0]
            plot_phi(Y_train_phi[0].detach().cpu().numpy(),
                     phi_pred[0].detach().cpu().numpy(),
                     t_train, i_train, epoch)
            compare_all_turbines(model, t=42)

            # 挑一个测试数据出来看看是怎么个事
            t_test, i_test = meta_test[-1]
            plot_phi(Y_test_phi[-1].detach().cpu().numpy(),
                     model(X_test)[-1].detach().cpu().numpy(),
                     t_test, i_test, f"{epoch} test")
            compare_all_turbines(model, t=100 - 42)

    Path = "./model/inflow_angle_model_FullConstraints.pth"
    save_model(model, Path)

    # 测试预测函数
    print("\n测试预测函数:")
    example_model = example_usage(Path)

    # 或者直接使用预测函数
    print("\n直接预测示例:")
    model_loaded = load_model(Path, Nr=len(R_), device=device)
    Ft, T, LFt, LTs = predict_fan_forces(model_loaded, 13.0, 1.26, 10.0)
    print(f"Ft={Ft:.2f} N, T={T:.2f} Nm")