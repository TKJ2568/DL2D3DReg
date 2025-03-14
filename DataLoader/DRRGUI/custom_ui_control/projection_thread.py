import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from .CommunicatedSignal import Communicate

# 自定义 QThread 子类
class ProjectionThread(QThread):
    # 定义信号，用于传递投影结果或错误信息
    projection_finished = pyqtSignal(np.ndarray)  # 投影成功，传递图像数据
    projection_failed = pyqtSignal(str)  # 投影失败，传递错误信息

    def __init__(self, projector):
        super().__init__()
        self.projector = projector
        self.im_update_signal = Communicate()  # 图像更新信号

        self._stop_flag = False  # 标志变量，用于指示是否需要停止当前操作
        self._latest_request = None  # 最新的请求参数
        self._lock = QMutex()  # 线程锁，确保线程安全

    @property
    def d_s2p(self):
        return self.projector.d_s2p

    @d_s2p.setter
    def d_s2p(self, value):
        self.projector.d_s2p = value

    @property
    def im_sz(self):
        return self.projector.im_sz

    @im_sz.setter
    def im_sz(self, value):
        if value[0] >= 512 or value[1] >= 512:
            self.projector.projector.set_project_mode("matrix_perspective")
        else:
            self.projector.projector.set_project_mode("circular_perspective")
        self.projector.im_sz = value

    def request_projection(self, init_pose, rota_noise, tran_noise):
        """接收新的投影请求"""
        with QMutexLocker(self._lock):  # 使用锁保护对共享资源的访问
            self._latest_request = (init_pose, rota_noise, tran_noise)
            self._stop_flag = True  # 设置标志，通知当前线程停止运行

        if not self.isRunning():
            self.start()  # 如果线程未运行，则启动线程

    def run(self):
        while True:
            with QMutexLocker(self._lock):  # 使用锁保护对共享资源的访问
                if self._latest_request is None:
                    break  # 如果没有新的请求，退出线程

                # 获取最新的请求参数
                init_pose, rota_noise, tran_noise = self._latest_request
                self._latest_request = None  # 清空请求
                self._stop_flag = False  # 重置停止标志

                # 设置初始姿态
                self.projector.set_rotation(init_pose[0], init_pose[1], init_pose[2])

                # 执行耗时的投影操作
                img = self.projector.project(
                    rota_noise, tran_noise, mode='not_display', save=False, save_path=None
                )

                # 检查是否需要停止
                if self._stop_flag:
                    continue  # 如果有新的请求，跳过当前结果

                # 发送图像更新信号
                self.im_update_signal.array_update_signal.emit(img)

            if self._stop_flag:
                continue  # 如果有新的请求，重新开始处理

            break  # 处理完当前请求后退出线程