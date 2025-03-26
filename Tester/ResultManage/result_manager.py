import os

from utils.dire_check import manage_folder
from .result_logger import ResultLogger
from .result_visualizer import ResultVisualizer


class ResultManager:
    def __init__(self, test_name_list, **kwargs):
        self.result_loggers = {}
        self.init_result_logger(kwargs.get("model_name_list", []), test_name_list)
        self.result_visualizer = ResultVisualizer(test_name_list, **kwargs)
        self.save_path = kwargs.get("save_path", "./")

    def init_result_logger(self, model_name_list, test_name_list):
        for model_name in model_name_list:
            self.result_loggers[model_name] = ResultLogger(model_name, test_name_list)

    def add_one_result(self, model_name, test_name, result):
        self.result_loggers[model_name].add_result(test_name, result)

    def visualize_result_and_save(self):
        # 保存图片
        manage_folder(self.save_path, "test_results")
        self.result_visualizer.visualize_results(self.result_loggers)
        save_path = os.path.join(self.save_path, "test_results")
        for logger in self.result_loggers.values():
            logger.save_result(save_path)