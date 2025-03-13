import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from .CommunicatedSignal import Communicate

# 自定义 QThread 子类
class ProjectionThread(QThread):
    # 定义信号，用于传递投影结果或错误信息
    projection_finished = pyqtSignal(np.ndarray)  # 投影成功，传递图像数据
    projection_failed = pyqtSignal(str)  # 投影失败，传递错误信息

    def __init__(self, projector):
        super().__init__()
        self.projector = projector
        self.init_pose = None
        self.rota_noise = None
        self.tran_noise = None
        self.im_update_signal = Communicate()  # 图像更新信号

    def run(self):
        self.projector.set_rotation(self.init_pose[0], self.init_pose[1], self.init_pose[2])
        # 执行耗时的投影操作
        img = self.projector.project(
            self.rota_noise, self.tran_noise, mode='not_display', save=False, save_path=None
        )
        self.im_update_signal.array_update_signal.emit(img)  # 发送图像更新信号