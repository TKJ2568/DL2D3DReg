import numpy as np
from skimage.metrics import structural_similarity as ssim
from Tester.CustomMetrics.abstract_metrics import AbstractMetrics


class ImageSSIMTest(AbstractMetrics):
    def __init__(self, **kwargs):
        """
        初始化ImageSSIMTest类

        参数:
        kwargs -- 包含必要参数的字典，必须包含'projector'键
        """
        self.name = "ImageSSIM"
        self.projector = kwargs['projector']
        # 可以添加SSIM特有的参数，如data_range, win_size等
        self.data_range = kwargs.get('data_range', 255)  # 图像像素值范围，默认255
        self.win_size = kwargs.get('win_size', 7)  # 滑动窗口大小，默认7
        self.multichannel = kwargs.get('multichannel', True)  # 是否为多通道图像，默认True

    def __call__(self, tru, pre):
        """
        计算两幅图像的结构相似性指数(SSIM)

        参数:
        tru -- 真实值数据
        pre -- 预测值数据

        返回:
        ssim_value -- 结构相似性指数，范围[-1, 1]，1表示完全相同
        """
        # 使用projector生成图像
        tru_img = self.projector.project(tru[:3], tru[3:], mode='not_display')
        pre_img = self.projector.project(tru[:3], pre[3:], mode='not_display')

        # 确保图像数据类型为float
        tru_img = tru_img.astype(np.float64)
        pre_img = pre_img.astype(np.float64)

        # 计算SSIM
        ssim_value = ssim(
            tru_img,
            pre_img,
            data_range=self.data_range,
            win_size=self.win_size,
            multichannel=self.multichannel,
            channel_axis=-1  # 对于新版本skimage
        )

        return ssim_value