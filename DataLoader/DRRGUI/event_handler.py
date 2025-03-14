import os

import cv2
import numpy as np
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from DataLoader.DRR import Projector
from .custom_ui_control import messageBox, ProjectionThread
from .init_para import InitPara
from .utils.format_transform import array2pixmap


class EventHandler(InitPara):
    def __init__(self, ui_config, drr_config):
        super().__init__(ui_config, drr_config)
        self.voxel_load_clip_ui.voxel_load_signal.connect(self.load_ct_voxel_file)
        self.splitter.splitterMoved.connect(self.update_drr_pixmap)
        # section 连接位姿改变信号
        self.init_rx_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.init_ry_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.init_rz_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.noise_rx_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.noise_ry_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.noise_rz_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.noise_tx_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.noise_ty_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        self.noise_tz_custom_slider.value_changed_signal.simple_signal.connect(self.update)
        # 其他投影参数修改响应
        self.d_s2p_text.returnPressed.connect(self.update)
        self.im_sz_text.returnPressed.connect(self.update)
        self.im_sz_down_btn.clicked.connect(self.im_sz_down)
        self.im_sz_up_btn.clicked.connect(self.im_sz_up)
        # 刷新，重置和保存当前图像
        self.refresh_btn.clicked.connect(self.update)
        self.reset_btn.clicked.connect(self.reset)
        self.save_im_btn.clicked.connect(self.save_im)

    def load_ct_voxel_file(self, file_path):
        projector = Projector(directory=file_path)
        self.projector_thread = ProjectionThread(projector)
        self.projector_thread.im_update_signal.array_update_signal.connect(self.update_im)
        self.update()
        self.voxel_load_clip_ui.voxel_file_load_control.set_line_edit_content(file_path)

    def update(self):
        if self.projector_thread is not None:
            # 设置初始投影参数
            d_s2p = int(self.d_s2p_text.text())
            im_sz = eval(self.im_sz_text.text())
            self.projector_thread.d_s2p = d_s2p
            self.projector_thread.im_sz = im_sz
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
            init_pose = [init_pose_x, init_pose_y, init_pose_z]
            self.projector_thread.request_projection(init_pose, rota_noise, tran_noise)
        else:
            messageBox("请先加载CT数据")
    def update_im(self, im):
        if im is None:
            messageBox("投影失败, 请检查参数设置")
            return
        self.img = im
        self.drr_pixmap = array2pixmap(im)
        self.update_drr_pixmap()

    def update_drr_pixmap(self):
        if self.drr_pixmap is None:
            return
        self.drr_img_label.setPixmap(
            self.drr_pixmap.scaled(self.drr_img_label.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))  # 在label上显示图片

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.update_drr_pixmap()

    def im_sz_down(self):
        im_sz = eval(self.im_sz_text.text())
        if im_sz[0] < self.drr_config['min_im_sz'][0] or im_sz[1] < self.drr_config['min_im_sz'][1]:
            messageBox("图像尺寸过小, 无法缩小")
            return
        im_sz = (im_sz[0] // 2, im_sz[1] // 2)
        self.im_sz_text.setText(str(im_sz))
        self.update()

    def im_sz_up(self):
        im_sz = eval(self.im_sz_text.text())
        im_sz = (im_sz[0] * 2, im_sz[1] * 2)
        if im_sz[0] > self.drr_config['max_im_sz'][0] or im_sz[1] > self.drr_config['max_im_sz'][1]:
            messageBox("图像尺寸过大, 无法放大")
            return
        self.im_sz_text.setText(str(im_sz))
        self.update()

    def reset(self):
        self.init_rx_custom_slider.set_value(self.drr_config['init_pose'][0])
        self.init_ry_custom_slider.set_value(self.drr_config['init_pose'][1])
        self.init_rz_custom_slider.set_value(self.drr_config['init_pose'][2])
        self.noise_rx_custom_slider.set_value(0)
        self.noise_ry_custom_slider.set_value(0)
        self.noise_rz_custom_slider.set_value(0)
        self.noise_tx_custom_slider.set_value(0)
        self.noise_ty_custom_slider.set_value(0)
        self.noise_tz_custom_slider.set_value(0)
        self.d_s2p_text.setText(str(self.drr_config['d_s2p']))
        self.im_sz_text.setText(str(self.drr_config['im_sz']))
        self.update()

    def save_im(self):
        if self.img is None:
            messageBox("请先生成DRR图像")
            return
        # 获取当前工作目录
        current_directory = os.getcwd()
        # 设置默认文件名
        default_filename = os.path.join(current_directory, "im.png")
        # 弹出文件保存对话框，设置默认路径和文件名
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存DRR图像", default_filename, "PNG(*.png);;JPG(*.jpg);;BMP(*.bmp)"
        )
        if file_path == "":
            return  # 用户取消保存
        try:
            # 使用 OpenCV 保存图像
            cv2.imwrite(file_path, self.img)
            QMessageBox.information(self, "成功", f"图像已保存到: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图像时出错: {str(e)}")