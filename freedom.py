# freedom.py (标准化向量化版)
import pandas as pd
import numpy as np
import torch
from scipy.interpolate import interp1d

# ======================================================
# 配置数据包
# ======================================================
PACKAGE = 2
print(f"正在读取模拟数据表附件{PACKAGE}")
DF_THETA_WF1 = pd.read_excel(f"DATA{PACKAGE}/Theta100.xlsx").values
DF_V1_WF1 = pd.read_excel(f"DATA{PACKAGE}/V100.xlsx").values
DF_OMEGA_WF1 = pd.read_excel(f"DATA{PACKAGE}/Omega_r100.xlsx").values
DF_OMEGAG_WF1 = pd.read_excel(f"DATA{PACKAGE}/Omega_f100.xlsx").values
DF_TSHAFT_WF1 = pd.read_excel(f"DATA{PACKAGE}/Tshaft100.xlsx").values
DF_FT_WF1 = pd.read_excel(f"DATA{PACKAGE}/Ft100.xlsx").values
DF_PREF_WF1 = pd.read_excel(f"DATA{PACKAGE}/Pref100.xlsx").values

# ======================================================
# 叶片几何及气动扭转
# ======================================================
DF_C = pd.read_excel("du40.xlsx", sheet_name='C').values
R_ = DF_C[:, 0]        # 径向位置
C_ = DF_C[:, 1]        # 弦长
DR_ = DF_C[:, 2]       # 叶素长度
AT_ = DF_C[:, 3]       # 气动扭转 [deg]

# ======================================================
# 空气动力学常量
# ======================================================
class Const:
    rho = 1.29
    B = 3
    Gr = 97
    J_Gen = 534.116
    k = 8.67637e8
    c = 6.215e6
    eta = 0.944

# ======================================================
# 升阻力系数表插值
# ======================================================
def _load_airfoil(sheet):
    DF = pd.read_excel("du40.xlsx", sheet_name=sheet).values
    alpha = DF[:,0]/180*np.pi
    cl = DF[:,1]
    cd = DF[:,2]
    return alpha, cl, cd

ALPHA_DU40, CL_DU40, CD_DU40 = _load_airfoil('DU40')
ALPHA_DU35, CL_DU35, CD_DU35 = _load_airfoil('DU35')
ALPHA_DU30, CL_DU30, CD_DU30 = _load_airfoil('DU30')
ALPHA_DU25, CL_DU25, CD_DU25 = _load_airfoil('DU25')
ALPHA_DU21, CL_DU21, CD_DU21 = _load_airfoil('DU21')
ALPHA_NACA64, CL_NACA64, CD_NACA64 = _load_airfoil('NACA64')

CL_interp_DU40 = interp1d(ALPHA_DU40, CL_DU40, kind='cubic', bounds_error=False, fill_value="extrapolate")
CD_interp_DU40 = interp1d(ALPHA_DU40, CD_DU40, kind='cubic', bounds_error=False, fill_value="extrapolate")
CL_interp_DU35 = interp1d(ALPHA_DU35, CL_DU35, kind='cubic', bounds_error=False, fill_value="extrapolate")
CD_interp_DU35 = interp1d(ALPHA_DU35, CD_DU35, kind='cubic', bounds_error=False, fill_value="extrapolate")
CL_interp_DU30 = interp1d(ALPHA_DU30, CL_DU30, kind='cubic', bounds_error=False, fill_value="extrapolate")
CD_interp_DU30 = interp1d(ALPHA_DU30, CD_DU30, kind='cubic', bounds_error=False, fill_value="extrapolate")
CL_interp_DU25 = interp1d(ALPHA_DU25, CL_DU25, kind='cubic', bounds_error=False, fill_value="extrapolate")
CD_interp_DU25 = interp1d(ALPHA_DU25, CD_DU25, kind='cubic', bounds_error=False, fill_value="extrapolate")
CL_interp_DU21 = interp1d(ALPHA_DU21, CL_DU21, kind='cubic', bounds_error=False, fill_value="extrapolate")
CD_interp_DU21 = interp1d(ALPHA_DU21, CD_DU21, kind='cubic', bounds_error=False, fill_value="extrapolate")
CL_interp_NACA64 = interp1d(ALPHA_NACA64, CL_NACA64, kind='cubic', bounds_error=False, fill_value="extrapolate")
CD_interp_NACA64 = interp1d(ALPHA_NACA64, CD_NACA64, kind='cubic', bounds_error=False, fill_value="extrapolate")
C_interp = interp1d(R_, C_, kind='cubic', bounds_error=False, fill_value="extrapolate")

# ======================================================
# 升阻力系数计算 (向量化)
# alpha: [B,Nr] torch.Tensor
# 返回 cl, cd: [B,Nr]
# ======================================================
def calc_cl_cd(angle):
    """
    angle: ndarray, shape=(Nr, Nt)
    返回:
        cl, cd: ndarray, shape=(Nr, Nt)
    """
    cl = np.zeros_like(angle)
    cd = np.zeros_like(angle)

    # 圆柱1
    cl[0:2] = 0
    cd[0:2] = 0.5
    # 圆柱2
    cl[2] = 0
    cd[2] = 0.35
    # DU40
    cl[3] = CL_interp_DU40(angle[3])
    cd[3] = CD_interp_DU40(angle[3])
    # DU35
    cl[4:6] = CL_interp_DU35(angle[4:6])
    cd[4:6] = CD_interp_DU35(angle[4:6])
    # DU30
    cl[6] = CL_interp_DU30(angle[6])
    cd[6] = CD_interp_DU30(angle[6])
    # DU25
    cl[7:9] = CL_interp_DU25(angle[7:9])
    cd[7:9] = CD_interp_DU25(angle[7:9])
    # DU21
    cl[9:11] = CL_interp_DU21(angle[9:11])
    cd[9:11] = CD_interp_DU21(angle[9:11])
    # NACA64
    cl[11:] = CL_interp_NACA64(angle[11:])
    cd[11:] = CD_interp_NACA64(angle[11:])

    return cl, cd


# ======================================================
# sigma 计算
# ======================================================
def calc_sigma(r, device=None):
    c = C_interp(r)
    sigma = Const.B * c / (2*np.pi*r)
    t = torch.tensor(sigma, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


# ======================================================
# BEM a,b迭代 (向量化)
# v1, omega, theta: [B]
# 返回 a,b: [B,Nr]
# ======================================================
def calc_ab(r, v1, omega, theta, eps):
    """
    计算诱导速度因子 a, b
    r: 叶片径向坐标, 1D ndarray
    v1: 迎风速度, 1D ndarray, 每个风机一个值
    omega: 转速, 1D ndarray
    theta: 桨距角, 1D ndarray
    eps: 迭代允许误差
    返回:
        a, b: ndarray, shape=(len(r), len(v1))
    """
    r = np.array(r)
    v1 = np.array(v1)
    omega = np.array(omega)
    theta = np.array(theta)

    Nr = len(r)
    Nt = len(v1)
    a = np.zeros((Nr, Nt))
    b = np.zeros((Nr, Nt))

    error = 5000
    j = 0
    while error > eps and j < 20:
        j += 1
        a0, b0 = a.copy(), b.copy()

        # 入流速度
        inflow_vel = (1 - a) * v1[np.newaxis, :]  # [Nr,Nt]
        rotation_term = (1 + b) * omega[np.newaxis, :] * r[:, np.newaxis]  # [Nr,Nt]
        phi = np.arctan2(inflow_vel, rotation_term)

        alpha = phi - (theta[np.newaxis, :] + AT_[:, np.newaxis]) / 180 * np.pi
        cl, cd = calc_cl_cd(alpha)
        sigma = Const.B * C_[:, np.newaxis] / 2 / np.pi / r[:, np.newaxis]

        cn = cl * np.cos(phi) + cd * np.sin(phi)
        ct = cl * np.sin(phi) - cd * np.cos(phi)

        a = 1 / (4 * np.sin(phi)**2 / sigma / cn + 1)
        b = 1 / (4 * np.sin(phi) * np.cos(phi) / sigma / ct - 1)

        error = np.sum((a - a0)**2 + (b - b0)**2)

    return a, b


# ======================================================
# Ft/T 计算 (向量化)
# phi: [B,Nr], V/Omega/Theta: [B]
# 返回 Ft,T: [B]
# ======================================================
def calc_ftp_from_phi(phi, V, Omega, Theta):
    B,Nr = phi.shape
    device = phi.device
    r_t = torch.tensor(R_, dtype=torch.float32, device=device).view(1,Nr)
    dr_t = torch.tensor(DR_, dtype=torch.float32, device=device).view(1,Nr)
    C_t = torch.tensor(C_, dtype=torch.float32, device=device).view(1,Nr)
    AT_t = torch.tensor(AT_, dtype=torch.float32, device=device).view(1,Nr)

    V = V.view(B,1)
    Omega = Omega.view(B,1)
    Theta = Theta.view(B,1)

    W = torch.sqrt((V*torch.sin(phi))**2 + (Omega*r_t*torch.cos(phi))**2)
    alpha = phi - (Theta + AT_t)/180*np.pi
    cl, cd = calc_cl_cd(alpha)
    cn = cl*torch.cos(phi) + cd*torch.sin(phi)
    ct = cl*torch.sin(phi) - cd*torch.cos(phi)

    dFt = 0.5*Const.rho*Const.B*W**2*C_t*cn*dr_t
    dT = 0.5*Const.rho*Const.B*W**2*C_t*ct*r_t*dr_t

    Ft = torch.sum(dFt, dim=1)
    T = torch.sum(dT, dim=1)
    return Ft, T


