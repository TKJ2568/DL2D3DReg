import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm as gauss_norm


def cal_voxel_mse_loss_tensor(cube, rt1, rt2, mode='average', weight=1):
    """
    计算两个变换矩阵 rt1 和 rt2 对 cube 的变换后的顶点之间的 MSE 损失。
    :param cube: 体素的顶点坐标 (N, 4)，最后一列为权重
    :param rt1: 第一个变换矩阵 (batch_size, 4, 4)
    :param rt2: 第二个变换矩阵 (batch_size, 4, 4)
    :param mode: 计算模式，'average' 返回平均距离，'max' 返回最大距离
    :param weight: 权重
    :return: 损失值
    """
    batch_size = rt1.shape[0]  # 获取批次大小

    # 将 cube 的最后一列从权重改为 1（齐次坐标要求）
    cube_homogeneous = cube.clone()  # 复制 cube，避免修改原始数据
    cube_homogeneous[:, 3] = 1.0  # 将最后一列设置为 1

    # 将 cube 转换为齐次坐标 (N, 4) -> (N, 4, 1)
    cube_homogeneous = cube_homogeneous.unsqueeze(-1)  # (N, 4, 1)

    # 将 cube 扩展到批次维度 (N, 4, 1) -> (batch_size, N, 4, 1)
    cube_homogeneous = cube_homogeneous.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, N, 4, 1)

    # 对 cube 应用变换矩阵 rt1 和 rt2
    new_pos1 = torch.matmul(rt1.unsqueeze(1), cube_homogeneous).squeeze(-1)  # (batch_size, N, 4)
    new_pos2 = torch.matmul(rt2.unsqueeze(1), cube_homogeneous).squeeze(-1)  # (batch_size, N, 4)

    # 计算顶点之间的距离（仅计算前 3 维，忽略齐次坐标的最后一维）
    distance = torch.linalg.norm(new_pos1[:, :, :3] - new_pos2[:, :, :3], dim=2)  # (batch_size, N)

    # 应用权重（使用 cube 的原始权重列）
    distance = distance * weight * cube[:, 3]  # (batch_size, N)

    # 根据模式返回损失
    if mode == 'average':
        return torch.mean(distance, dim=1)  # (batch_size,)
    else:
        return torch.max(distance, dim=1)[0]  # (batch_size,)


def add_noise_tensor(angle_noise, trans_noise, rot_cen):
    """
    添加噪声到变换矩阵中。
    :param angle_noise: 角度噪声 (batch_size, 3)
    :param trans_noise: 平移噪声 (batch_size, 3)
    :param rot_cen: 旋转中心 (3,)
    :return: 加噪后的变换矩阵 (batch_size, 4, 4)
    """
    batch_size = angle_noise.shape[0]
    device = angle_noise.device

    # 从体素到旋转中心的变换矩阵
    ct2rot_cen = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    ct2rot_cen[:, :3, 3] = -rot_cen

    # 从旋转中心到加噪后的旋转中心的变换矩阵
    pi = torch.tensor(np.pi, device=device)
    R = _euler_angles2rot_matrix_tensor(angle_noise[:, 0] * pi / 180,
                                        angle_noise[:, 1] * pi / 180,
                                        angle_noise[:, 2] * pi / 180)  # (batch_size, 3, 3)
    t_noise = torch.cat([trans_noise, torch.ones(batch_size, 1, device=device)], dim=1)  # (batch_size, 4)

    rot_cen_2noised_cen1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    rot_cen_2noised_cen1[:, :3, :3] = R

    rot_cen_2noised_cen2 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    rot_cen_2noised_cen2[:, :, 3] = t_noise

    rot_cen_2noised_cen = torch.matmul(rot_cen_2noised_cen2, rot_cen_2noised_cen1)  # (batch_size, 4, 4)

    # 从加噪后的旋转中心到体素的变换矩阵
    noised_cen2noised_ct = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    noised_cen2noised_ct[:, :3, 3] = rot_cen

    # 从体素到加噪后的体素的变换矩阵
    ct2noised_ct = torch.matmul(noised_cen2noised_ct, torch.matmul(rot_cen_2noised_cen, ct2rot_cen))  # (batch_size, 4, 4)
    return ct2noised_ct


def _euler_angles2rot_matrix_tensor(theta_x, theta_y, theta_z):
    """
    将欧拉角转换为旋转矩阵（支持批量操作）。
    :param theta_x: X 轴旋转角度 (batch_size,)
    :param theta_y: Y 轴旋转角度 (batch_size,)
    :param theta_z: Z 轴旋转角度 (batch_size,)
    :return: 旋转矩阵 (batch_size, 3, 3)
    """
    batch_size = theta_x.shape[0]
    device = theta_x.device

    # 构造旋转矩阵
    cx, sx = torch.cos(theta_x), torch.sin(theta_x)
    cy, sy = torch.cos(theta_y), torch.sin(theta_y)
    cz, sz = torch.cos(theta_z), torch.sin(theta_z)

    rot_x = torch.zeros(batch_size, 3, 3, device=device)
    rot_x[:, 0, 0] = 1
    rot_x[:, 1, 1] = cx
    rot_x[:, 1, 2] = sx
    rot_x[:, 2, 1] = -sx
    rot_x[:, 2, 2] = cx

    rot_y = torch.zeros(batch_size, 3, 3, device=device)
    rot_y[:, 0, 0] = cy
    rot_y[:, 0, 2] = -sy
    rot_y[:, 1, 1] = 1
    rot_y[:, 2, 0] = sy
    rot_y[:, 2, 2] = cy

    rot_z = torch.zeros(batch_size, 3, 3, device=device)
    rot_z[:, 0, 0] = cz
    rot_z[:, 0, 1] = sz
    rot_z[:, 1, 0] = -sz
    rot_z[:, 1, 1] = cz
    rot_z[:, 2, 2] = 1

    return torch.matmul(rot_z, torch.matmul(rot_y, rot_x))


def init_vertex(vertex_size, interval_num):
    """
    根据体素大小生成等间隔采样的点云。
    :param vertex_size: 体素大小 (3,)
    :param interval_num: 每条边上的采样点数
    :return: 点云坐标和权重 (N, 4)
    """
    interval = vertex_size / interval_num
    x, y, z = np.mgrid[0.5 * interval[0]:vertex_size[0]:interval[0],
               0.5 * interval[1]:vertex_size[1]:interval[1],
               0.5 * interval[2]:vertex_size[2]:interval[2]]
    centers = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    # 计算权重
    r = np.sqrt((vertex_size[0] / 2) ** 2 + (vertex_size[1] / 2) ** 2)
    xy = centers[:, :2]
    center = np.array([vertex_size[0] / 2, vertex_size[1] / 2])
    distances = np.linalg.norm(xy - center[None, :], axis=1)
    weight = gauss_norm.pdf(distances / r * 3)
    weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))  # 归一化

    pt_matrix = np.zeros((centers.shape[0], 4))
    pt_matrix[:, :3] = centers
    pt_matrix[:, 3] = weight
    return pt_matrix


class VmLoss(nn.Module):
    def __init__(self, voxel_size, interval_num, rot_cen, is_cuda):
        super(VmLoss, self).__init__()
        self.rot_cen = torch.tensor(rot_cen, dtype=torch.float32)
        self.cube = torch.tensor(init_vertex(voxel_size, interval_num), dtype=torch.float32)
        if is_cuda:
            self.cube = self.cube.cuda()

    def forward(self, output, target):
        """
        计算 VM 损失。
        :param output: 预测的位姿 (batch_size, 6)
        :param target: 真实的位姿 (batch_size, 6)
        :return: 损失值
        """
        # 获取预测值和真实值
        pre_alpha, pre_beta, pre_theta, pre_tx, pre_ty, pre_tz = output.unbind(dim=1)
        tru_alpha, tru_beta, tru_theta, tru_tx, tru_ty, tru_tz = target.unbind(dim=1)

        # 计算变换矩阵
        rt1 = self._get_rt_tensor(tru_alpha, tru_beta, tru_theta, tru_tx, tru_ty, tru_tz)
        rt2 = self._get_rt_tensor(pre_alpha, pre_beta, pre_theta, pre_tx, pre_ty, pre_tz)

        # 计算损失
        loss = cal_voxel_mse_loss_tensor(self.cube, rt1, rt2)
        return loss.mean()

    def _get_rt_tensor(self, alpha, beta, theta, tx, ty, tz):
        """
        计算变换矩阵。
        :param alpha: 旋转角度 (batch_size,)
        :param beta: 旋转角度 (batch_size,)
        :param theta: 旋转角度 (batch_size,)
        :param tx: 平移 (batch_size,)
        :param ty: 平移 (batch_size,)
        :param tz: 平移 (batch_size,)
        :return: 变换矩阵 (batch_size, 4, 4)
        """
        angle_noise = torch.stack([alpha, beta, theta], dim=1)
        trans_noise = torch.stack([tx, ty, tz], dim=1)
        return add_noise_tensor(angle_noise, trans_noise, self.rot_cen)