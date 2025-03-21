"""
记录网络信息
1. 网络名称
2. 网络的训练参数
3. 网络的训练时间
4. 网络使用的优化器
5. 网络使用的损失函数
6. 网络训练结果
"""
import copy
import json
import os
from ModelManager.utils import Logger
from utils.dire_check import manage_folder

# 定义颜色代码
class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"  # 重置颜色

class NetworkInfo:
    def __init__(self, init_info: dict):
        self.name = init_info['name']
        self.info = init_info

        self.train_log = Logger()
        self.eval_log = Logger()

    def set_train_info(self, train_info:dict):
        self.info.update(**train_info)

    def save_network_info(self, save_log_config):
        """
        将 NetworkInfo 类的所有成员变量保存到指定文件夹下的 name 子文件夹中。
        :param save_log_config: str, 保存文件的根路径
        """
        # 创建 name 子文件夹
        save_path = save_log_config['save_path']
        name_folder = os.path.join(save_path, self.name)
        manage_folder(save_path, self.name)

        # 保存为 JSON 文件
        save_file_path = os.path.join(str(name_folder), "network_info.json")
        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, indent=4, ensure_ascii=False)

        # 保存训练日志
        train_fig_config = save_log_config['train_fig_config']
        save_name = train_fig_config["save_name"] + ".html"
        train_fig_config["title"] = self.name + train_fig_config["title_suffix"]
        train_fig_config["save_path"] = os.path.join(str(name_folder), save_name)
        train_fig_config["y_label_loss"] = str(self.info['loss_func'])
        train_fig_config["y_label_precision"] = str(self.info['eval_metric'])
        self.train_log.visualization(train_fig_config)

        save_name = train_fig_config["save_name"] + ".csv"
        save_file_path = os.path.join(str(name_folder), save_name)
        self.train_log.save_to_csv(save_file_path)

        # 保存评估日志
        eval_fig_config = save_log_config['eval_fig_config']
        save_name = eval_fig_config["save_name"] + ".html"
        eval_fig_config["title"] = self.name + eval_fig_config["title_suffix"]
        eval_fig_config["save_path"] = os.path.join(str(name_folder), save_name)
        eval_fig_config["y_label_loss"] = str(self.info['loss_func'])
        eval_fig_config["y_label_precision"] = str(self.info['eval_metric'])
        self.eval_log.visualization(eval_fig_config)

        save_name = eval_fig_config["save_name"] + ".csv"
        save_file_path = os.path.join(str(name_folder), save_name)
        self.eval_log.save_to_csv(save_file_path)

        print(f"{Colors.YELLOW}{self.name} 已保存到 {name_folder}{Colors.RESET}")