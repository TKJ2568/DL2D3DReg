import os

import numpy as np

from utils.dire_check import manage_folder
from .abnormal_analyse import AbnormalAnalyser
from .result_logger import ResultLogger
from .result_visualizer import ResultVisualizer


class ResultManager:
    def __init__(self, init_pose, projector, test_name_list, **kwargs):
        self.result_loggers = {}
        self.init_result_logger(kwargs.get("model_name_list", []), test_name_list)
        self.result_visualizer = ResultVisualizer(test_name_list, **kwargs)
        self.abnormal_analyser = AbnormalAnalyser(init_pose,projector, **kwargs)
        self.save_path = kwargs.get("save_path", "./")

    def init_result_logger(self, model_name_list, test_name_list):
        for model_name in model_name_list:
            self.result_loggers[model_name] = ResultLogger(model_name, test_name_list)

    def add_one_metric_result(self, model_name, test_name, result):
        self.result_loggers[model_name].add_metric_result(test_name, result)

    def add_one_real_result(self, model_name, truth, predict):
        self.result_loggers[model_name].add_true_and_predict(truth, predict)

    def visualize_result_and_save(self):
        # 统计每个模型的异常分数
        for logger in self.result_loggers.values():
            normalized_result = np.array(logger.get_normalized_result())
            self.abnormal_analyser.abnormal_score.append(np.sum(normalized_result, axis=0))
        # 保存图片
        manage_folder(self.save_path, "test_results")
        self.result_visualizer.visualize_metrics_results(self.result_loggers)
        self.abnormal_analyser.abnormal_visualize()
        save_path = os.path.join(self.save_path, "test_results")
        for logger in self.result_loggers.values():
            logger.save_result(save_path)
