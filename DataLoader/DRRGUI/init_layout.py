from PyQt6 import QtWidgets
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow, QSpacerItem

from .custom_ui_control import VoxelLoadClipUI, CustomSlider
from .ui.drr_gui import Ui_MainWindow


def init_custom_slider(config, refresh_rate):
    name = config["name"]
    min_value = config["min_value"]
    max_value = config["max_value"]
    init_value = config["default_value"]
    return CustomSlider(name, min_value, max_value, init_value, refresh_rate)


class InitUILayout(QMainWindow, Ui_MainWindow):
    def __init__(self, ui_config):
        super().__init__()
        self.setupUi(self)
        # 设置窗口标题和图标
        self.setWindowTitle("投射可视化界面")
        self.setWindowIcon(QIcon("res/X光.ico"))
        # section 1 体素导入以及初始化
        icon_path = r"res/open_file.png"
        self.voxel_load_clip_ui = VoxelLoadClipUI(icon_path, self.horizontalLayout_10)
        # 添加初始位姿可调节控件
        refresh_rate = ui_config["refresh_rate"]
        self.init_rx_custom_slider = init_custom_slider(ui_config["init_rx_slider"], refresh_rate)
        self.init_ry_custom_slider = init_custom_slider(ui_config["init_ry_slider"], refresh_rate)
        self.init_rz_custom_slider = init_custom_slider(ui_config["init_rz_slider"], refresh_rate)
        self.init_pos_layout.addWidget(self.init_rx_custom_slider)
        self.init_pos_layout.addWidget(self.init_ry_custom_slider)
        self.init_pos_layout.addWidget(self.init_rz_custom_slider)
        init_pose_spacer = QSpacerItem(20, 160, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.init_pos_layout.addSpacerItem(init_pose_spacer)
        # 添加噪声可调节的控件
        self.noise_rx_custom_slider = init_custom_slider(ui_config["noise_rx_slider"], refresh_rate)
        self.noise_ry_custom_slider = init_custom_slider(ui_config["noise_ry_slider"], refresh_rate)
        self.noise_rz_custom_slider = init_custom_slider(ui_config["noise_rz_slider"], refresh_rate)
        self.noise_tx_custom_slider = init_custom_slider(ui_config["noise_tx_slider"], refresh_rate)
        self.noise_ty_custom_slider = init_custom_slider(ui_config["noise_ty_slider"], refresh_rate)
        self.noise_tz_custom_slider = init_custom_slider(ui_config["noise_tz_slider"], refresh_rate)
        self.noise_layout.addWidget(self.noise_rx_custom_slider)
        self.noise_layout.addWidget(self.noise_ry_custom_slider)
        self.noise_layout.addWidget(self.noise_rz_custom_slider)
        self.noise_layout.addWidget(self.noise_tx_custom_slider)
        self.noise_layout.addWidget(self.noise_ty_custom_slider)
        self.noise_layout.addWidget(self.noise_tz_custom_slider)
        noise_spacer = QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.noise_layout.addSpacerItem(noise_spacer)

    def showEvent(self, a0):
        self.resize(1200, 400)

