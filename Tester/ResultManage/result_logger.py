"""
一个模型记录对应的测试结果数据
"""
import os
import pandas as pd
import numpy as np


class ResultLogger:
    def __init__(self, model_name, test_names):
        self.model_name = model_name
        self.result_dict = {}
        self.init_result_dict(test_names)

    def init_result_dict(self, test_names):
        for test_name in test_names:
            self.result_dict[test_name] = []

    def add_result(self, test_name, result):
        self.result_dict[test_name].append(result)

    def get_normalized_result(self):
        """
        返回各个测试结果，并对每项数据除以该项数据的平均值
        :return:
        """
        results = []
        for result_list in self.result_dict.values():
            arr = np.array(result_list)
            # arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
            arr = arr / np.mean(arr)
            results.append(np.around(arr.astype(np.float64), 2).tolist())
        return results

    def get_mean_result(self):
        """
        返回各个测试结果的平均值
        :return:
        """
        results = []
        for result_list in self.result_dict.values():
            arr = np.array(result_list)
            mean_result = np.mean(arr)
            results.append(mean_result.tolist())
        return results

    def save_result(self, save_path):
        """
        将测试结果保存为 CSV 文件，文件名为模型名称。
        """
        # 构建保存路径
        file_name = f"{self.model_name}.csv"
        file_path = os.path.join(save_path, file_name)

        # 构建 DataFrame
        data = {test_name: results for test_name, results in self.result_dict.items()}
        df = pd.DataFrame(data)

        # 保存为 CSV 文件
        df.to_csv(file_path, index=False)



