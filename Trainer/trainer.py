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


def get_optimizer_list(model_list, config):
    optimizer_list = []
    if config['optimizer'] == 'adam':
        for model in model_list:
            optimizer_list.append(optim.Adam(model.parameters(), lr=config['lr']))
        return optimizer_list

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
        # if is_cuda:
        #     self.model_manager.model_group_list = [model.cuda() for model in self.model_manager.model_group_list]
        # self.optimizer_list = get_optimizer_list(self.model_manager.model_group_list, config)
        self.loss_function = get_loss_function(config, is_cuda)
        self.metric_function = get_metric_function(config)
        self.is_cuda = is_cuda
        self.config = config
        self.train_trail()

    def train_once(self, model, optimizer):
        model.train()
        images, labels = next(iter(self.data_loader.train_loader))
        output = model(images)
        # 从归一化的值恢复到真实值
        labels = self.data_loader.label_transformer.label2real(labels)
        output = self.data_loader.label_transformer.label2real(output)
        loss = self.loss_function(output, labels)
        # 计算MPD
        mpd = self.metric_function(labels, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), mpd

    def eval_once(self, model):
        model.eval()
        images, labels = next(iter(self.data_loader.eval_loader))
        with torch.no_grad():
            output = model(images)
            # 从归一化的值恢复到真实值
            labels = self.data_loader.label_transformer.label2real(labels)
            output = self.data_loader.label_transformer.label2real(output)
            loss = self.loss_function(output, labels)
            # 计算MPD
            mpd = self.metric_function(labels, output)
            return loss.item(), mpd

    def train_and_eval_one_model(self, model: NetworkFrame, optimizer):
        # 初始化 tqdm 进度条
        epoch_bar = tqdm(range(self.epoch), desc="Training", unit="epoch")
        # 记录训练开始时间
        start_time = time.time()
        final_loss = 10000
        for epoch in epoch_bar:
            train_loss, train_mpd = self.train_once(model, optimizer)
            final_loss = train_loss
            model.network_info.train_log.add_entry(epoch, train_loss, train_mpd)
            epoch_bar.set_postfix({"Train Loss": train_loss, "Train MPD": train_mpd})
            if epoch % self.config['eval_interval'] == 0  and epoch != 0:
                eval_loss, eval_mpd = self.eval_once(model)
                model.network_info.eval_log.add_entry(epoch, eval_loss, eval_mpd)
                epoch_bar.write(f"Eval at Epoch {epoch}: Loss: {eval_loss}, MPD: {eval_mpd}")
            model.model_saver.save_checkpoint(model, optimizer, epoch, train_loss)
        # 保存最后的模型
        model.model_saver.save_model(model, final_loss)
        # 记录训练结束时间
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def train(self):
        for model, optimizer in zip(self.model_manager.model_group_list, self.optimizer_list):
            training_time = self.train_and_eval_one_model(model, optimizer)
            train_info = {"train_epoch": self.epoch,
                          "train_batch_size": self.data_loader.batch_size,
                          "train_time": training_time,
                          "optimizer": self.config['optimizer'],
                          "loss_func": self.config['loss_function'],
                          "eval_metric": self.config['eval_metric']}
            assert isinstance(model, NetworkFrame)
            model.network_info.set_train_info(train_info)
            model.network_info.save_network_info(self.config['save_log_config'])

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

