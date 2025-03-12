from DataLoader.DRR import ct_projector
from .dicom_manager import *

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 投影体生成类
class Projector:
    def __init__(self, pixel_status="raw_pixel", cropped="whole_body", directory=None):
        super().__init__()
        # 投影相关参数
        self.d_s2p = 800
        self.im_sz = [512, 512]
        self.pan_sz = [240, 240]
        self.alpha = 0
        self.beta = 0
        self.theta = 0
        self.filenames = None
        self.light_pos_voxel = None
        self.Rt = None
        self.is_augment = False
        # 添加dicom管理类
        self.dicom_manager = Manager(pixel_status=pixel_status, cropped=cropped)
        self.dicom_manager.open_folder(directory)
        self.projector = ct_projector.CTProjector(self.dicom_manager.ct_vox, self.dicom_manager.vox_space)

    # 体素文件所在文件夹读取
    def load_ct_images(self, directory):
        self.dicom_manager.open_folder(directory).open_folder(directory)
        # plot_3d(self.ct_vox, 400)
        print("CT voxel shape, type and spacing are {0}, {1}, {2}".
              format(self.dicom_manager.ct_vox.shape, self.dicom_manager.ct_vox.dtype, self.dicom_manager.vox_space))

    def show_ct_info(self):
        """
        打印导入的CT体素的相关信息
        :return:
        """
        print("CT voxel shape, type and spacing are {0}, {1}, {2}".
              format(self.dicom_manager.ct_vox.shape, self.dicom_manager.ct_vox.dtype, self.dicom_manager.vox_space))

    # 设置旋转参数
    def set_rotation(self, angle_x, angle_y, angle_z):
        self.alpha = angle_x
        self.beta = angle_y
        self.theta = angle_z

    # 生成投影图像
    # 在生成投影图像的过程中，输入平移分量并生成平移矩阵
    def project(self, angle_noise, trans_noise, mode='not_display', save=False, save_path=None):
        # print("d_s2p: {}, im_sz: {}, pan_sz: {}, theta_z: {}".format(self.d_s2p, self.im_sz, self.pan_sz, self.theta))
        # Method 1: using the class CTProjector:
        _a_arm = self.projector.set_A_arm(self.d_s2p, self.im_sz, self.pan_sz)
        # 通过改变旋转中心，改变投影图的显示效果
        origin_rot_cen = self.projector.get_rot_cen()
        isOK, self.Rt = self.projector.set_Rt(self.theta, self.beta, self.alpha, d_s2c=self.d_s2p / 2)
        self.projector.set_rot_cen(origin_rot_cen)
        if mode == 'get_rm&arm':
            return self.Rt, _a_arm
        if isOK:
            # time0 = time.time()
            # 投影之前加载
            angle_noise = np.pi * angle_noise / 180
            xray_img = self.projector.project(noise=(angle_noise, trans_noise))
            # print("Projection time: ", time.time() - time0)
            xray_img = norm_image(xray_img)
            if save:
                im = Image.fromarray(xray_img)
                # C:\Users\adminTKJ\Desktop\MainProject\ctguide - probe\data\ct_img
                im.save(save_path+"/x{0}_y{1}_z{2}.png".
                        format(self.alpha, self.beta, self.theta))
            if mode == 'display':
                plt.figure()
                plt.imshow(xray_img, cmap='gray')
                plt.show()
            return xray_img.reshape(self.im_sz[0], self.im_sz[1])


if __name__ == "__main__":
    # 标准正位 0 270 90或者0 90 270
    # 标准侧位 0 270 0或者0 90 180
    # CT_path = r"D:\Project\CT_pinvisual\new_ZDQ"
    projector = Projector(directory=r"C:\Users\adminTKJ\Desktop\MainProject\CT_投影方式研究\CT_data\spineV3_raw")
    projector.show_ct_info()
    # projector.load_ct_images(directory=CT_path)
    projector.set_rotation(0, 180, 180)
    # ["circular_perspective", "matrix_perspective", "orthogonal"]
    projector.projector.set_project_mode("circular_perspective")
    rota_noise = np.array([0, 0, 0], dtype=np.float32)
    tran_noise = np.array([0, 0, 0],  dtype=np.float32)
    projector.project(rota_noise, tran_noise, mode='display', save=False)
