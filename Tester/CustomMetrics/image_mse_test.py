import numpy as np

from Tester.CustomMetrics.abstract_metrics import AbstractMetrics


class ImageMSETest(AbstractMetrics):
    def __init__(self, **kwargs):
        self.name = "ImageMSE"
        self.projector = kwargs['projector']

    def __call__(self, tru, pre):
        tru_img = self.projector.project(tru[:3], tru[3:], mode='not_display')
        pre_img = self.projector.project(tru[:3], pre[3:], mode='not_display')
        mse = np.mean((tru_img - pre_img) ** 2)
        return mse