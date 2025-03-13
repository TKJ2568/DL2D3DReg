"""
自定义的滑动条，该滑动条由三部分组成
1. label: 显示当前修改项
2. slider: 滑动条
3. spinbox: 显示当前值
移动滑块时spinbox同步变化. 且当值改变时，在延时后发出更新信号
"""
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QTimer

from DataLoader.DRRGUI.custom_ui_control.CommunicatedSignal import Communicate


class CustomSlider(QtWidgets.QWidget):
    def __init__(self, label_text, min_value, max_value, init_value, interval):
        super(CustomSlider, self).__init__()

        # 初始化控件
        self.label = QtWidgets.QLabel(label_text)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setValue(init_value)

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setMinimum(min_value)
        self.spinbox.setMaximum(max_value)
        self.spinbox.setValue(init_value)

        # 连接信号和槽
        self.slider.valueChanged.connect(self.update_value_by_slider)
        self.spinbox.valueChanged.connect(self.update_value_by_spinbox)

        # 当参数改变时，发出更新信号
        self.value_changed_timer = QTimer(singleShot=True, interval=interval)
        self.value_changed_timer.timeout.connect(self.emit_value_changed_signal)
        self.value_changed_signal = Communicate()

        # 设置布局
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.spinbox)

    def update_value_by_slider(self):
        """当滑块的值改变时，更新 SpinBox 的值"""
        slider_value = self.slider.value()
        self.spinbox.setValue(slider_value)
        self.start_value_changed_timer()

    def update_value_by_spinbox(self):
        """当 SpinBox 的值改变时，更新滑块的值"""
        spinbox_value = self.spinbox.value()
        self.slider.setValue(spinbox_value)
        self.start_value_changed_timer()

    def value(self):
        """获取当前的值"""
        return self.slider.value()

    def set_value(self, value):
        """设置滑块和 SpinBox 的值"""
        self.slider.setValue(value)
        self.spinbox.setValue(value)

    def start_value_changed_timer(self):
        """开始计时器"""
        if self.value_changed_timer.isActive():
            self.value_changed_timer.stop()
        self.value_changed_timer.start()

    def emit_value_changed_signal(self):
        """发出更新信号"""
        self.value_changed_signal.simple_signal.emit()