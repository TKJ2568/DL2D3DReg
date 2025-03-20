import numpy as np

def generate_uv_coordinates(im_sz):
    """
    生成 UV 坐标矩阵。
    参数:
        im_sz: 图像的高度和宽度，例如 [128, 128]。
    返回:
        uv: 一个 [2, H, W] 的矩阵，第一个通道是 u，第二个通道是 v。
    """
    H, W = im_sz
    # 生成网格坐标
    x = np.linspace(0, 1, W)  # 水平方向 (u)
    y = np.linspace(0, 1, H)  # 垂直方向 (v)
    u, v = np.meshgrid(x, y)  # 生成网格

    # 将 u 和 v 堆叠成一个 [2, H, W] 的矩阵
    uv = np.stack([u, v], axis=0)
    return uv

def expand_uv_to_batch(uv, batch_size):
    """
    将 UV 坐标扩展到批次维度。
    参数:
        uv: UV 坐标矩阵，形状为 [2, H, W]。
        batch_size: 批次大小。
    返回:
        uv_batch: 扩展后的 UV 坐标矩阵，形状为 [batch_size, 2, H, W]。
    """
    uv_batch = np.tile(uv[np.newaxis, ...], (batch_size, 1, 1, 1))
    return uv_batch

# 线性变换回0-255
def norm_image(image, min_v=None, max_v=None, bits=None):
    if min_v is None:
        min_v = np.min(image)
    if max_v is None:
        max_v = np.max(image)
    if bits is None:
        return 255 * (image - min_v) / (max_v - min_v)
    else:
        return (2 ** bits - 1) * (image - min_v) / (max_v - min_v)