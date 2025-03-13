# 定义一个带有自定义信号的类
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal


class Communicate(QObject):
    simple_signal = pyqtSignal()
    str_update_signal = pyqtSignal(str)
    two_str_update_signal = pyqtSignal(str, str)
    bool_update_signal = pyqtSignal(str, bool)
    list_update_signal = pyqtSignal(list)
    array_update_signal = pyqtSignal(np.ndarray)