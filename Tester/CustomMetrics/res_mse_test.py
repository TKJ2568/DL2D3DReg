import numpy as np

from Tester.CustomMetrics.abstract_metrics import AbstractMetrics


class ResultMSETest(AbstractMetrics):
    def __init__(self):
        self.name = "ResultMSE"

    def __call__(self, tru, pre):
        """
        计算均方误差(Mean Squared Error)

        参数:
        tru -- 真实值(ground truth)，可以是numpy数组或列表
        pre -- 预测值(predicted values)，形状应与tru相同

        返回:
        mse -- 均方误差值
        """
        # 将输入转换为numpy数组以确保计算一致性
        tru = np.array(tru)
        pre = np.array(pre)

        # 检查输入形状是否一致
        if tru.shape != pre.shape:
            raise ValueError(f"形状不匹配: tru形状{tru.shape}, pre形状{pre.shape}")

        # 计算均方误差
        squared_errors = (tru - pre) ** 2
        mse = np.mean(squared_errors)

        return mse