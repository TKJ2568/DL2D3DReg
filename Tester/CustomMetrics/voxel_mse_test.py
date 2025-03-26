import copy

import numpy as np
from scipy.stats import norm as gauss_norm
from Tester.CustomMetrics.abstract_metrics import AbstractMetrics

def add_noise(angle_noise=np.array([0, 0, 0]), trans_noise=np.array([0, 0, 0]), rot_cen=None):
    """

    :param angle_noise: 角度噪声
    :param trans_noise: 位移噪声
    :param rot_cen: 旋转中心
    :return:
    """
    # 从体素到旋转中心
    ct2rot_cen = np.diag([1, 1, 1, 1]).astype(np.float32)
    ct2rot_cen[:3, 3] = -rot_cen
    # 从旋转中心到加噪后的旋转中心
    # 1 添加旋转噪声，
    R = euler_angles2rot_matrix(angle_noise[0] * np.pi / 180,
                                angle_noise[1] * np.pi / 180,
                                angle_noise[2] * np.pi / 180)
    # 添加平移噪声
    t_noise = np.hstack((trans_noise, np.array([1]))).T
    rot_cen_2noised_cen1 = np.diag([1, 1, 1, 1]).astype(np.float32)
    rot_cen_2noised_cen2 = np.diag([1, 1, 1, 1]).astype(np.float32)
    rot_cen_2noised_cen1[:3, :3] = R
    rot_cen_2noised_cen2[:, 3] = t_noise
    # print(rot_cen_2noised_cen1)
    # print(rot_cen_2noised_cen2)
    rot_cen_2noised_cen = rot_cen_2noised_cen2 @ rot_cen_2noised_cen1
    # 从加噪后的旋转中心到转动后的体素
    noised_cen2noised_ct = np.diag([1, 1, 1, 1]).astype(np.float32)
    noised_cen2noised_ct[:3, 3] = rot_cen
    # 从体素到转动后的体素
    ct2noised_ct = noised_cen2noised_ct @ rot_cen_2noised_cen @ ct2rot_cen
    return ct2noised_ct

def euler_angles2rot_matrix(theta_x, theta_y, theta_z):
    """
    转动角度需要输入为弧度
    :param theta_x: rotation angle along X axis
    :param theta_y: rotation angle along Y axis
    :param theta_z: rotation angle along Z axis
    :return: rotation matrix
    """
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)
    rot_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])

    cy = np.cos(theta_y)
    sy = np.sin(theta_y)
    rot_y = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])

    cz = np.cos(theta_z)
    sz = np.sin(theta_z)
    rot_z = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])

    return np.matmul(rot_z, np.matmul(rot_y, rot_x))

def norm(datas, min_v=None, max_v=None):
    if min_v is None:
        min_v = np.min(datas)
    if max_v is None:
        max_v = np.max(datas)
    return np.round((datas - min_v) / (max_v - min_v), 5)


def init_vertex(vertex_size, interval_num):
    """
    根据体素实际的大小，在其中等间隔采点。对于每个切片来说，按高斯分布，边缘的权重大，中间的权重小。
    :param vertex_size: 体素实际的大小
    :param interval_num:从体素中每条边上取多少个点，默认值的意思是会从体素中均分出100*100*100个小长方体
    :return:
    """
    # 生成小正方体中心的坐标网格
    # 使用np.mgrid生成三个一维数组，代表x, y, z坐标，然后通过reshape和transpose转换成(3, 1000)的形状
    interval = vertex_size / interval_num
    x, y, z = np.mgrid[0.5 * interval[0]:vertex_size[0]:interval[0],
              0.5 * interval[1]:vertex_size[1]:interval[1],
              0.5 * interval[2]:vertex_size[2]:interval[2]]

    # 将三维坐标网格展平为一维数组，并组合成(3, 1000)的矩阵
    centers = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    # 首先计算切片内切圆的半径
    r = np.sqrt((vertex_size[0] / 2) ** 2 + (vertex_size[1] / 2) ** 2)
    # 沿z轴方向计算每个点在xy平面到旋转中心的距离
    xy = centers[:, :2]
    center = np.array([vertex_size[0] / 2, vertex_size[1] / 2])
    distances = np.linalg.norm(xy - center[None, :], axis=1)
    weight = distances / r
    # 对权重高斯映射
    weight = gauss_norm.pdf(weight * 3)
    # 权重归一化
    weight = norm(weight)
    pt_matrix = np.zeros((centers.shape[0], 4))
    pt_matrix[:, :3] = centers
    pt_matrix[:, 3] = weight

    return pt_matrix

class VoxelMSETest(AbstractMetrics):
    def __init__(self, **kwargs):
        self.name = 'VoxelMSE'
        self.voxel_size = np.array(kwargs.get('voxel_size'))
        self.rot_cen = np.array(self.voxel_size)/2
        self.interval_num = np.array(kwargs.get('interval_num'))
        self.d_s2c = kwargs.get('d_s2c')
        self.cube = init_vertex(self.voxel_size, self.interval_num)
        self.cube_homogeneous = copy.deepcopy(self.cube)
        self.cube_homogeneous[:, 3] = 1

    def __call__(self, tru, pre):
        tru_rt = add_noise(angle_noise=tru[:3], trans_noise=tru[3:], rot_cen=self.rot_cen)
        pre_rt = add_noise(angle_noise=pre[:3], trans_noise=pre[3:], rot_cen=self.rot_cen)
        new_pos1 = np.dot(tru_rt, self.cube_homogeneous.T).T
        new_pos2 = np.dot(pre_rt, self.cube_homogeneous.T).T
        distance = np.linalg.norm((new_pos1 - new_pos2), axis=1)
        distance = distance * self.cube[:, 3]
        return np.mean(distance)

