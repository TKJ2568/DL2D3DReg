from .get_xray import *
from scipy.spatial.transform import Rotation as R


def add_noise(angle_noise=np.array([0, 0, 0]), trans_noise=np.array([0, 0, 0]), rot_cen=None):
    """

    :param angle_noise: 角度噪声
    :param trans_noise: 位移噪声
    :param rot_cen: 旋转中心
    :return:
    """
    # 从体素到旋转中心
    # print("旋转中心\n", rot_cen)
    ct2rot_cen = np.diag([1, 1, 1, 1]).astype(np.float32)
    ct2rot_cen[:3, 3] = -rot_cen
    # print("从体素到旋转中心\n", ct2rot_cen)
    # 从旋转中心到加噪后的旋转中心
    # 1 添加旋转噪声
    R = _euler_angles2rot_matrix(angle_noise[0], angle_noise[1], angle_noise[2])
    # 添加平移噪声
    t_noise = np.hstack((trans_noise, np.array([1]))).T
    rot_cen_2noised_cen1 = np.diag([1, 1, 1, 1]).astype(np.float32)
    rot_cen_2noised_cen2 = np.diag([1, 1, 1, 1]).astype(np.float32)
    rot_cen_2noised_cen1[:3, :3] = R
    rot_cen_2noised_cen2[:, 3] = t_noise
    # print(rot_cen_2noised_cen1)
    # print(rot_cen_2noised_cen2)
    rot_cen_2noised_cen = rot_cen_2noised_cen2 @ rot_cen_2noised_cen1
    # print("从旋转中心到加噪后的旋转中心\n", rot_cen_2noised_cen)
    # 从加噪后的旋转中心到转动后的体素
    noised_cen2noised_ct = np.diag([1, 1, 1, 1]).astype(np.float32)
    noised_cen2noised_ct[:3, 3] = rot_cen
    # print("从加噪后的旋转中心到转动后的体素\n", noised_cen2noised_ct)
    # 从体素到转动后的体素
    ct2noised_ct = noised_cen2noised_ct @ rot_cen_2noised_cen @ ct2rot_cen
    # print("从体素到转动后的体素\n", ct2noised_ct)
    return ct2noised_ct


def project_from_ct3(Rt, A_arm, ct_vox, vox_space=np.array([1, 1, 1]), im_sz=None, interp=1, project_mode=None):
    """
    this function computer x-ray projection image from CT voxel data with projection parameters:

    :param Rt: transform matrix from x-ray source coordinates to CT coordinates
    :param A_arm: projection matrix A from x-ray source coordinates to image plane
    :param ct_vox: 3D CT voxel data
    :param vox_space: voxel spacing of CT data (mm), if Rt has been voxelization, vox_space can be omitted
    :param im_sz: size of X-ray image (pixel)
    :param interp: interpolation for integral along rays
    :param project_mode: ["circular_perspective", "matrix_perspective", "orthogonal"]
    :return: x-ray image (im_sz[1], im_sz[0])
    """

    if im_sz is None:
        # 如果没有给定像平面的大小，则默认是体素中心点的两倍
        im_sz = np.array([A_arm[0, 2] * 2 + 1, A_arm[1, 2] * 2 + 1]).astype(np.int32)

    o_src, rays, max_axis = _compute_src_rays2(im_sz, A_arm, Rt, vox_space)
    # time0 = time.time()
    slice_list = np.linspace(1 / (2 * interp) - 0.5, ct_vox.shape[max_axis] - 0.5 - 1 / (2 * interp),
                             num=ct_vox.shape[max_axis] * interp, dtype=np.float32)
    vox_sum_col = np.zeros((ct_vox.shape[max_axis], im_sz[0] * im_sz[1]))

    # 初始化每条射线的权重为1
    ray_weighs = 1
    if project_mode == "circular_perspective":
        vox_sum_col, ray_weighs = compute_cross_voxel(slice_list=slice_list,
                                                      max_axis=max_axis,
                                                      o_src=o_src,
                                                      rays=rays,
                                                      ct_vox=ct_vox,
                                                      vox_sum_col=vox_sum_col)
    if project_mode == "matrix_perspective":
        vox_sum_col, ray_weighs = compute_cross_voxel_matrix(slices=slice_list,
                                                             max_axis=max_axis,
                                                             o_src=o_src,
                                                             rays=rays,
                                                             ct_vox=ct_vox,
                                                             vox_sum_col=vox_sum_col)
    if project_mode == "orthogonal":
        vox_sum_col, ray_weighs = orthogonal_projection(slice_list=slice_list,
                                                        max_axis=max_axis,
                                                        o_src=o_src,
                                                        rays=rays,
                                                        ct_vox=ct_vox,
                                                        vox_sum_col=vox_sum_col)
    # print("main time for projection: ", time.time() - time0)
    x_ray = (vox_sum_col * ray_weighs / (ct_vox.shape[max_axis] * interp)).reshape(im_sz[1], im_sz[0])
    return x_ray


def get_default_A_arm(d_s2p, im_sz, pan_sz):
    """
    :param d_s2p: distance between x-ray source and 2D imaging panel (mm)
    :param im_sz: x-ray image size on panel (pixel)
    :param pan_sz: real panel size (mm)
    :return: projection matrix A from x-ray source coordinate system to image plane
    """
    return np.array([[-d_s2p * im_sz[0] / pan_sz[0], 0, (im_sz[0] - 1) / 2],
                     [0, d_s2p * im_sz[1] / pan_sz[1], (im_sz[1] - 1) / 2],
                     [0, 0, 1]])


def get_default_Rt(d_s2p, vox_shape, vox_space, theta, d_s2c=None):
    """
    :param d_s2p: distance between x-ray source and 2D imaging panel (mm)
    :param vox_shape: shape of the 3D CT voxel data (voxel)
    :param vox_space: voxel spacing of the CT data (mm)
    :param theta: rotation angle along z axis of the CT data
    :param d_s2c: distance between x-ray source and center of the voxel
    :return: transform matrix from x-ray source coordinate to CT coordinate
    """

    # center of CT data in the CT coordinates system (mm)
    ct_cen = ((np.array(vox_shape, dtype=np.float32) - 1) * np.array(vox_space)) / 2
    # distance between x-ray source and CT center
    if d_s2c is None:
        d_s2c = d_s2p - np.sqrt(
            (vox_shape[0] * vox_space[0]) ** 2 + (vox_shape[1] * vox_space[1]) ** 2) / 2

    theta_rad = theta * np.pi / 180
    # translation vector i.e. the 3D coordinates of the x-ray source in the CT coordinates system
    t_vec = np.array([ct_cen[0] + d_s2c * np.cos(theta_rad), ct_cen[1] - d_s2c * np.sin(theta_rad), ct_cen[2]])
    # compute rotation matrix from Euler angles (theta_x, theta_y, theta_z) with the default rotation order
    rot_matrix = _euler_angles2rot_matrix(0, np.pi / 2, theta_rad)
    return np.column_stack((rot_matrix, t_vec))


def get_Rt(rot_cen, d_s2c, theta, beta, alpha):
    """
    compute the Rt matrix from x-ray source coordinate to CT coordinate

    :param rot_cen: rotation center of C-Arm (mm) in CT coordinate
    :param d_s2c: distance between x-ray source and rot_cen (mm)
    :param theta: rotation angle along z axis of the CT data
    :param beta: angle between z axis of CT and z axis of x-ray source, i.e. rotation angle along y-axis of x-ray source
    :param alpha: rotation angle of image plane along z axis of the C-Arm
    :return: transform matrix from x-ray source coordinate to CT coordinate
    """
    rot_z = _euler_angles2rot_matrix(0, 0, alpha * np.pi / 180)
    # compute rotation matrix from Euler angles (theta_x, theta_y, theta_z) with the default rotation order
    rot_matrix = _euler_angles2rot_matrix(0, np.pi * beta / 180, theta * np.pi / 180) @ rot_z
    # translation vector i.e. the 3D coordinates of the x-ray source in the CT coordinates system
    t_vec = np.array(rot_matrix @ np.mat([0, 0, -d_s2c]).T).squeeze() + np.array(rot_cen)
    return np.column_stack((rot_matrix, t_vec)).astype(np.float32)


def _euler_angles2rot_matrix(theta_x, theta_y, theta_z):
    """
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


def _compute_src_rays2(im_sz, A_arm, Rt, vox_space):
    """
    the function is used to compute 3D coordinates of the pixels on panel in voxel coordinate system

    :param im_sz: size of the X-ray image (pixel)
    :param A_arm: projection matrix A from x-ray source coordinate system to image plane
    :param Rt: transform matrix from x-ray source coordinates to CT coordinates
    :param vox_space: voxel spacing of the CT data (mm)
    :return: src_pos: 3D voxel coordinates of the x-ray source (in the voxel coordinates system)
        rays: difference vectors between 3D voxel coordinates of the pixels in the x-ray image (panel) and src_pos；
    """
    pixel_u, pixel_v = np.meshgrid(np.arange(0, im_sz[0]), np.arange(0, im_sz[1]))
    pixel_uv1 = np.vstack((pixel_u.ravel(), pixel_v.ravel(), np.ones(pixel_u.size)))
    R_vox = (Rt[:, 0:3] / vox_space.reshape(-1, 1)).astype(np.float32)
    o_src = (Rt[:, 3] / vox_space).astype(np.float32)
    rays_unit = ((R_vox @ np.linalg.inv(A_arm)) @ pixel_uv1).T.astype(np.float32)
    ray_cen = ((R_vox @ np.linalg.inv(A_arm)) @ [0, 0, 1]).astype(np.float32)
    # the principal (max) projection axis, max_axis = 0, 1, 2 means accumulation along x, y, z axis, respectively
    max_axis = np.argmax(np.abs(ray_cen)).astype(int)
    # compute the weights of every rays

    return o_src, rays_unit, max_axis


# 用于测试相关参数是否合理
def test_para(vox_shape, vox_space, Rt, d_s2p):
    # Test whether d_s2c is outside a reasonable range
    vert_ct = [[0, 0, 0],
               [vox_shape[0], 0, 0],
               [0, vox_shape[1], 0],
               [0, 0, vox_shape[2]],
               [vox_shape[0], vox_shape[1], 0],
               [vox_shape[0], 0, vox_shape[2]],
               [0, vox_shape[1], vox_shape[2]],
               [vox_shape[0], vox_shape[1], vox_shape[2]]
               ] * np.array(vox_space)
    vert_src = (np.linalg.inv(Rt[:, 0:3])) @ (vert_ct - Rt[:, 3].T).T
    return np.max(vert_src[2, :]) < d_s2p and np.min(vert_src[2, :]) > 0, vert_src


class CTProjector:

    def __init__(self, ct_vox=None, vox_spa=None, theta=180):
        self._vox = ct_vox
        self._vox_spa = vox_spa

        # load default intrinsic paras. Undone: reading from configuration file
        self._pan_sz = [180, 180]
        self._im_sz = [512, 512]
        self._d_s2p = 800
        self._A_arm = get_default_A_arm(self._d_s2p, self._im_sz, self._pan_sz)

        # load default extrinsic paras
        # _rot_cen is the rotation center of C_Arm, which defaults to the center of CT (mm)
        self._rot_cen = ((np.array(self._vox.shape, dtype=np.float32) - 1) * np.array(self._vox_spa)) / 2
        self._theta = theta
        self._beta = 90
        # _alpha is the rotation angle of C-Arm along its z axis
        self._alpha = 0
        self._d_s2c = self._d_s2p - np.sqrt((self._vox.shape[0] * self._vox_spa[0]) ** 2 +
                                            (self._vox.shape[1] * self._vox_spa[1]) ** 2) / 2
        self._Rt = None
        self._x_ray = None

        # 投影模式
        self.project_mode = "circular_perspective"

    def get_rot_cen(self):
        return self._rot_cen

    def set_rot_cen(self, rot_cen):
        self._rot_cen = rot_cen

    def get_d_s2c(self):
        return self._d_s2c

    def reload_ct(self, ct_vox, vox_spa):
        self._vox = ct_vox
        self._vox_spa = vox_spa

    def set_A_arm(self, d_s2p=None, im_sz=None, pan_sz=None):
        """
        :param d_s2p: distance between x-ray source and 2D imaging panel (mm)
        :param im_sz: x-ray image size on panel (pixel)
        :param pan_sz: real panel size (mm)
        """
        if d_s2p is not None:
            self._d_s2p = d_s2p
        if pan_sz is not None:
            self._pan_sz = pan_sz
        if im_sz is not None:
            self._im_sz = im_sz
        self._A_arm = get_default_A_arm(self._d_s2p, self._im_sz, self._pan_sz).astype(np.float32)
        return self._A_arm

    def set_Rt(self, theta=None, beta=None, alpha=None, d_s2c=None, rot_cen=None):

        if theta is not None:
            self._theta = theta
        if beta is not None:
            self._beta = beta
        if alpha is not None:
            self._alpha = alpha
        if rot_cen is not None:
            self._rot_cen = rot_cen
        if d_s2c is not None:
            self._d_s2c = d_s2c

        self._Rt = get_Rt(self._rot_cen, self._d_s2c, self._theta, self._beta, self._alpha)
        # print("投影角参数：")
        # print(self._theta, self._beta, self._alpha)
        # print("投影矩阵为：")
        # print(self._Rt)
        isOK, ver_src = test_para(self._vox.shape, self._vox_spa, self._Rt, self._d_s2p)
        if not isOK:
            print('voxel is out side c-arm: d_s2c = {}, rot_cen = {}, ver_src: '.
                  format(self._d_s2c, self._rot_cen), ver_src)
        return isOK, self._Rt

    def set_project_mode(self, project_mode):
        """
        :param project_mode: ["circular_perspective", "matrix_perspective", "orthogonal"]
        :return:
        """
        self.project_mode = project_mode
        print("投影模式设置成功，当前投影模式为：", self.project_mode)

    def show_project_mode(self):
        print("当前投影模式为：", self.project_mode)

    def project(self, Rt=None, A_arm=None, noise=None):
        # this method computer x-ray projection image from CT voxel data with projection parameters:
        # 附加的tm为平移矩阵
        if Rt is not None:
            self._Rt = Rt
        if A_arm is not None:
            self._A_arm = A_arm
        # 投影过程中添加噪声
        if noise is not None:
            RT = np.vstack((self._Rt, [0, 0, 0, 1]))
            self._Rt = (add_noise(noise[0], noise[1], self._rot_cen) @ RT)[:3]
        _Rt_vox = self._Rt / self._vox_spa.reshape(-1, 1)
        self._x_ray = project_from_ct3(_Rt_vox, self._A_arm, self._vox, project_mode=self.project_mode)
        return self._x_ray

    def get_proj_para(self):
        vox_shape = self._vox.shape
        vert_ct = [[0, 0, 0],
                   [vox_shape[0], 0, 0],
                   [0, vox_shape[1], 0],
                   [0, 0, vox_shape[2]],
                   [vox_shape[0], vox_shape[1], 0],
                   [vox_shape[0], 0, vox_shape[2]],
                   [0, vox_shape[1], vox_shape[2]],
                   [vox_shape[0], vox_shape[1], vox_shape[2]]
                   ] * np.array(self._vox_spa)
        # vox_lines_id = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4],
        #                 [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]

        vert_uv1 = np.array([[0, 0, 1],
                             [0, self._im_sz[1], 1],
                             [self._im_sz[0], self._im_sz[1], 1],
                             [self._im_sz[0], 0, 1]]).T * self._d_s2p

        vert_im = (self._Rt[:, 0:3] @ np.linalg.inv(self._A_arm)) @ vert_uv1 + self._Rt[:, 3].reshape(-1, 1)
        # im_lines_id = [[0, 1], [1, 2], [2, 3], [3, 0]]
        return self._A_arm, self._Rt, self._rot_cen, vert_ct, vert_im.T
