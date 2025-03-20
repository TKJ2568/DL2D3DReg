"""
该函数设计出来用于判断两个正方体在经过RT变换之后，每个顶点之间的距离，并返回其最大值
"""
import numpy as np

def add_noise_batch(angle_noise, trans_noise, rot_cen=None):
    """
    支持批量输入的角度噪声和平移噪声
    :param angle_noise: 角度噪声，形状为 (batch_size, 3)
    :param trans_noise: 平移噪声，形状为 (batch_size, 3)
    :param rot_cen: 旋转中心，形状为 (3,)
    :return: 变换矩阵，形状为 (batch_size, 4, 4)
    """
    batch_size = angle_noise.shape[0]
    ct2rot_cen = np.tile(np.eye(4), (batch_size, 1, 1)).astype(np.float32)  # (batch_size, 4, 4)
    ct2rot_cen[:, :3, 3] = -rot_cen  # 从体素到旋转中心

    # 计算旋转矩阵
    R = euler_angles2rot_matrix_batch(angle_noise[:, 0] * np.pi / 180,
                                      angle_noise[:, 1] * np.pi / 180,
                                      angle_noise[:, 2] * np.pi / 180)  # (batch_size, 3, 3)

    # 添加平移噪声
    t_noise = np.hstack((trans_noise, np.ones((batch_size, 1))))  # (batch_size, 4)
    rot_cen_2noised_cen1 = np.tile(np.eye(4), (batch_size, 1, 1)).astype(np.float32)  # (batch_size, 4, 4)
    rot_cen_2noised_cen2 = np.tile(np.eye(4), (batch_size, 1, 1)).astype(np.float32)  # (batch_size, 4, 4)
    rot_cen_2noised_cen1[:, :3, :3] = R  # 旋转部分
    rot_cen_2noised_cen2[:, :, 3] = t_noise  # 平移部分

    # 组合旋转和平移
    rot_cen_2noised_cen = np.matmul(rot_cen_2noised_cen2, rot_cen_2noised_cen1)  # (batch_size, 4, 4)

    # 从加噪后的旋转中心到转动后的体素
    noised_cen2noised_ct = np.tile(np.eye(4), (batch_size, 1, 1)).astype(np.float32)  # (batch_size, 4, 4)
    noised_cen2noised_ct[:, :3, 3] = rot_cen  # 平移部分

    # 从体素到转动后的体素
    ct2noised_ct = np.matmul(noised_cen2noised_ct, np.matmul(rot_cen_2noised_cen, ct2rot_cen))  # (batch_size, 4, 4)

    return ct2noised_ct

def euler_angles2rot_matrix_batch(theta_x, theta_y, theta_z):
    """
    批量计算欧拉角到旋转矩阵
    :param theta_x: 绕 X 轴旋转角度，形状为 (batch_size,)
    :param theta_y: 绕 Y 轴旋转角度，形状为 (batch_size,)
    :param theta_z: 绕 Z 轴旋转角度，形状为 (batch_size,)
    :return: 旋转矩阵，形状为 (batch_size, 3, 3)
    """
    batch_size = theta_x.shape[0]
    R = np.zeros((batch_size, 3, 3))

    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cz, sz = np.cos(theta_z), np.sin(theta_z)

    # 绕 X 轴旋转
    R[:, 0, 0] = 1
    R[:, 1, 1] = cx
    R[:, 1, 2] = sx
    R[:, 2, 1] = -sx
    R[:, 2, 2] = cx

    # 绕 Y 轴旋转
    Ry = np.zeros((batch_size, 3, 3))
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = -sy
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = sy
    Ry[:, 2, 2] = cy

    # 绕 Z 轴旋转
    Rz = np.zeros((batch_size, 3, 3))
    Rz[:, 0, 0] = cz
    Rz[:, 0, 1] = sz
    Rz[:, 1, 0] = -sz
    Rz[:, 1, 1] = cz
    Rz[:, 2, 2] = 1

    # 组合旋转矩阵
    R = np.matmul(Rz, np.matmul(Ry, R))

    return R

def cube_max_distance_batch(rt1_batch, rt2_batch, voxel_size):
    """
    批量计算顶点之间的最大距离
    :param rt1_batch: 第一个 RT 矩阵的批量
    :param rt2_batch: 第二个 RT 矩阵的批量
    :param voxel_size: 体素大小(mm)
    :return: 每个 RT 矩阵对的最大顶点距离
    """
    # 定义正方体的顶点（单位：体素）
    cube = np.array([[-1, 1, -1, 1],  # 顶点 1
                     [-1, -1, -1, 1],  # 顶点 2
                     [1, 1, -1, 1],    # 顶点 3
                     [1, -1, -1, 1],   # 顶点 4
                     [-1, 1, 1, 1],    # 顶点 5
                     [-1, -1, 1, 1],   # 顶点 6
                     [1, 1, 1, 1],     # 顶点 7
                     [1, -1, 1, 1]]).astype(np.float32)   # 顶点 8

    # 将顶点坐标乘以 voxel_size，转换为实际物理尺寸
    cube[:, 0] *= voxel_size[0]  # 长
    cube[:, 1] *= voxel_size[1]  # 宽
    cube[:, 2] *= voxel_size[2]  # 高

    # 将顶点转换为齐次坐标
    cube_homogeneous = cube.T

    # 批量计算变换后的顶点位置
    new_pos1 = np.matmul(rt1_batch, cube_homogeneous)
    new_pos2 = np.matmul(rt2_batch, cube_homogeneous)

    # 计算顶点之间的距离
    distances = np.linalg.norm(new_pos1 - new_pos2, axis=1)

    # 返回每个 RT 矩阵对的最大顶点距离
    return np.max(distances, axis=1)

def calculate_max_distance(tru, pre, rot_cen, voxel_size):
    """
    输入一组位姿真值和估计，计算组中每一对的最大顶点距离，并返回最大顶点距离均值
    :param tru: 真实的位姿参数-欧拉角表示
    :param pre: 估计的位姿参数-欧拉角表示
    :param rot_cen: 旋转中心在体素中的位置
    :param voxel_size: 体素大小(mm)
    :return: 返回最大顶点距离的均值
    """
    try:
        tru = tru.detach().cpu().numpy()
        pre = pre.detach().cpu().numpy()
    except AttributeError:
        tru = tru.reshape(-1, 6)
        pre = pre.reshape(-1, 6)

    tru_rota = tru[:, :3]
    tru_trans = tru[:, 3:]
    pre_rota = pre[:, :3]
    pre_trans = pre[:, 3:]

    rt1 = add_noise_batch(tru_rota, tru_trans, rot_cen=rot_cen)
    rt2 = add_noise_batch(pre_rota, pre_trans, rot_cen=rot_cen)
    # 批量计算顶点距离
    max_dis = cube_max_distance_batch(rt1, rt2, voxel_size)

    # 返回最大顶点距离的均值
    return np.mean(max_dis)

class MaxPointDistance:
    def __init__(self, voxel_size, rot_cen):
        self.voxel_size = voxel_size
        self.rot_cen = rot_cen

    def __call__(self, tru, pre):
        return calculate_max_distance(tru, pre, self.rot_cen, self.voxel_size)
