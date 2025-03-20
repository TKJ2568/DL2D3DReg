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
        self.im_size = init_info['im_size']
        self.in_channels = init_info['in_channels']
        self.train_params = init_info['train_params']

        self.train_epoch = None
        self.train_batch_size = None
        self.train_time = None

        self.optimizer = None
        self.loss_func = None
        self.eval_metric = None

        self.train_log = Logger()
        self.eval_log = Logger()

    def set_train_info(self, train_info:dict):
        self.train_epoch = train_info['train_epoch']
        self.train_batch_size = train_info['train_batch_size']
        self.train_time = train_info['train_time']

        self.optimizer = train_info['optimizer']
        self.loss_func = train_info['loss_func']
        self.eval_metric = train_info['eval_metric']

    def save_network_info(self, save_log_config):
        """
        将 NetworkInfo 类的所有成员变量保存到指定文件夹下的 name 子文件夹中。
        :param save_log_config: str, 保存文件的根路径
        """
        # 创建 name 子文件夹
        save_path = save_log_config['save_path']
        name_folder = os.path.join(save_path, self.name)
        manage_folder(save_path, self.name)

        # 将成员变量保存为字典
        info_dict = {
            "name": self.name,
            "im_size": self.im_size,
            "in_channels": self.in_channels,
            "train_params": self.train_params,
            "train_epoch": self.train_epoch,
            "train_batch_size": self.train_batch_size,
            "train_time": self.train_time,
            "optimizer": str(self.optimizer),  # 将优化器对象转换为字符串
            "loss_func": str(self.loss_func),  # 将损失函数对象转换为字符串
            "eval_metric": str(self.eval_metric),  # 将评估指标对象转换为字符串
        }

        # 保存为 JSON 文件
        save_file_path = os.path.join(str(name_folder), "network_info.json")
        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(info_dict, f, indent=4, ensure_ascii=False)

        # 保存训练日志
        train_fig_config = save_log_config['train_fig_config']
        save_name = train_fig_config["save_name"] + ".html"
        train_fig_config["title"] = self.name + train_fig_config["title_suffix"]
        train_fig_config["save_path"] = os.path.join(str(name_folder), save_name)
        train_fig_config["y_label_loss"] = str(self.loss_func)
        train_fig_config["y_label_precision"] = str(self.eval_metric)
        self.train_log.visualization(train_fig_config)

        save_name = train_fig_config["save_name"] + ".csv"
        save_file_path = os.path.join(str(name_folder), save_name)
        self.train_log.save_to_csv(save_file_path)

        # 保存评估日志
        eval_fig_config = save_log_config['eval_fig_config']
        save_name = eval_fig_config["save_name"] + ".html"
        eval_fig_config["title"] = self.name + eval_fig_config["title_suffix"]
        eval_fig_config["save_path"] = os.path.join(str(name_folder), save_name)
        eval_fig_config["y_label_loss"] = str(self.loss_func)
        eval_fig_config["y_label_precision"] = str(self.eval_metric)
        self.eval_log.visualization(eval_fig_config)

        save_name = eval_fig_config["save_name"] + ".csv"
        save_file_path = os.path.join(str(name_folder), save_name)
        self.eval_log.save_to_csv(save_file_path)

        print(f"{Colors.YELLOW}{self.name} 已保存到 {name_folder}{Colors.RESET}")