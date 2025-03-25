import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from DataLoader import CustomDataLoader
from ModelManager import ModelManager
from ModelManager.network import NetworkGroup
from ModelManager.network.network_frame import NetworkFrame
from Trainer.CustomLoss import VmLoss
from Trainer.CustomMetric import MaxPointDistance

def get_loss_function(config, is_cuda):
    if config['loss_function'] == 'vm':
        vm_config = config['vm_config']
        voxel_size = np.array(vm_config['voxel_size'])
        interval_num = np.array(vm_config['interval_num'])
        rot_cen = np.array([voxel_size])/2
        return VmLoss(voxel_size, interval_num, rot_cen, is_cuda)

def get_metric_function(config):
    if config['eval_metric'] == 'MPD':
        MPD_config = config['MPD_config']
        voxel_size = np.array(MPD_config['voxel_size'])
        rot_cen = np.array(voxel_size)/2
        return MaxPointDistance(voxel_size, rot_cen)

class Trainer:
    def __init__(self, data_loader: CustomDataLoader, model_manager: ModelManager, config: dict, is_cuda):
        self.data_loader = data_loader
        self.model_manager = model_manager
        self.loss_function = get_loss_function(config, is_cuda)
        self.metric_function = get_metric_function(config)
        self.is_cuda = is_cuda
        self.config = config
        self.train_trail()

    def train_trail(self):
        # 创建模型组
        create_group_config = {"label_transformer": self.data_loader.label_transformer,
                               "dataset_tuple": (self.data_loader.train_dataset, self.data_loader.val_dataset)}
        self.model_manager.create_network_groups(**create_group_config)
        # 训练模型参数
        train_info = {"loss_function": self.loss_function,
                      "metric_function":self.metric_function,
                      "is_cuda": self.is_cuda,
                      "epoch": self.config['trail_epoch'],
                      "eval_interval": self.config['eval_interval'],
                      "trial_times": self.config['trial_times']}
        # 训练模型
        for model_group in self.model_manager.model_group_list:
            assert isinstance(model_group, NetworkGroup)
            model_group.initialize_train_parameters(**train_info)
            model_group.start_trial()
            # 从最优参数创建模型并训练
            train_best_params = {
                "epoch": self.config["final_epoch"],
                "loss_func": self.config['loss_function'],
                "eval_metric": self.config['eval_metric'],
                'save_log_config': self.config['save_log_config'],
            }
            model_group.train_from_best_params(**train_best_params)

