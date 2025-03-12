"""
负责图像显示过程中的格式转换问题
1 ndarray转为pixmap
"""
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem
from PIL import Image, ImageQt


def array2pixmap(array):
    image = Image.fromarray(array)
    pixmap = ImageQt.toqpixmap(image)
    return pixmap


def array2graphics_item(array):
    image = Image.fromarray(array)
    pixmap = ImageQt.toqpixmap(image)
    return QGraphicsPixmapItem(pixmap)
