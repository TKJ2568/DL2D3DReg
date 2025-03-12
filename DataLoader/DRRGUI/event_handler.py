import numpy as np
from PyQt6 import QtGui
from PyQt6.QtCore import Qt

from DataLoader.DRR import Projector
from .init_para import InitPara
from .utils.format_transform import array2pixmap


class EventHandler(InitPara):
    def __init__(self, ui_config, drr_config):
        super().__init__(ui_config, drr_config)
        self.voxel_load_clip_ui.voxel_load_signal.connect(self.load_ct_voxel_file)
        self.splitter.splitterMoved.connect(self.update_drr_img)

    def load_ct_voxel_file(self, file_path):
        self.projector = Projector(directory=file_path)
        self.update()
        self.voxel_load_clip_ui.voxel_file_load_control.set_line_edit_content(file_path)

    def update(self):
        if self.projector is not None:
            # 设置初始投影参数
            d_s2p = int(self.d_s2p_text.text())
            im_sz = eval(self.im_sz_text.text())
            self.projector.d_s2p = d_s2p
            self.projector.im_sz = im_sz
            # 设置位姿
            init_pose_x = self.init_rx_custom_slider.value()
            init_pose_y = self.init_ry_custom_slider.value()
            init_pose_z = self.init_rz_custom_slider.value()
            noise_rx = self.noise_rx_custom_slider.value()
            noise_ry = self.noise_ry_custom_slider.value()
            noise_rz = self.noise_rz_custom_slider.value()
            noise_tx = self.noise_tx_custom_slider.value()
            noise_ty = self.noise_ty_custom_slider.value()
            noise_tz = self.noise_tz_custom_slider.value()
            rota_noise = np.array([noise_rx, noise_ry, noise_rz], dtype=np.float32)
            tran_noise = np.array([noise_tx, noise_ty, noise_tz], dtype=np.float32)
            self.projector.set_rotation(init_pose_x, init_pose_y, init_pose_z)
            img = self.projector.project(rota_noise, tran_noise, mode='not_display', save=False, save_path=None)
            self.drr_pixmap = array2pixmap(img)
            self.update_drr_img()

    def update_drr_img(self):
        if self.drr_pixmap is None:
            return
        self.drr_img_label.setPixmap(
            self.drr_pixmap.scaled(self.drr_img_label.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))  # 在label上显示图片

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.update_drr_img()