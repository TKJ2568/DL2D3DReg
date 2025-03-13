# 定义一个带有自定义信号的类
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QWidget, QLineEdit, QPushButton, QFileDialog

from DataLoader.DRRGUI.custom_ui_control.CommunicatedSignal import Communicate


class ClipRangeUpdateSignal(QObject):
    signal = pyqtSignal(int, int)

class LineEditBtn(QWidget):
    def __init__(self, icon_size=15, icon_path=None):
        super().__init__()
        self.lineEdit = QLineEdit()
        self.lineEdit.setText("File not selected")
        self.btn = QPushButton()
        if icon_path is not None:
            self.btn.setIcon(QIcon(icon_path))
            self.btn.setIconSize(QSize(icon_size, icon_size))
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.lineEdit)
        self.layout.addWidget(self.btn)
        self.btn.clicked.connect(self.on_folder_select)
        # 定义文件文件夹导入完成后的信号
        self.folder_selected_signal = Communicate()

    def on_folder_select(self):
        # 打开文件选择对话框
        # TODO 当前就定义为我体素文件夹所在的路径，后续需要修改
        default_path = r"C:\Users\adminTKJ\Desktop\MainProject\CT_投影方式研究\CT_data\spine107_img_cropped"
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", default_path)
        if folder_path:
            self.folder_selected_signal.str_update_signal.emit(folder_path)

    def set_line_edit_content(self, content):
        # 更新标签显示选中的文件名
        self.lineEdit.setText(content)


class VoxelLoadClipUI:
    def __init__(self, icon_path, load_control_widget:QHBoxLayout):
        self.voxel_file_load_control = LineEditBtn(icon_path=icon_path)
        self.voxel_load_signal = self.voxel_file_load_control.folder_selected_signal.str_update_signal
        load_control_widget.addWidget(self.voxel_file_load_control)