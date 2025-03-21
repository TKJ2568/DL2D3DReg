"""
使用group的概念主要是对应同一个基本网络结构，不同参数的组合。
目前可以组合的超参数有：
1. optimizer = ['SGD', 'Adam', 'RMSprop']
2. learning_rate = [1e-5, 1e-1]
3. batch_size = [4, 8, 16, 32]
4. dropout_rate = [0.1, 0.5]
"""
import json
import os
import time

import optuna
import torch
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ModelManager.network.model_saver import ModelSaver
from ModelManager.network.network_frame import NetworkFrame
from ModelManager.network.network_info import NetworkInfo
from utils.dire_check import manage_folder


# 用于统计网络的可训练参数个数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class NetworkGroup:
    def __init__(self, general_config: dict, specific_config: dict, **kwargs):
        self.general_config = general_config
        self.specific_config = specific_config
        self.block = kwargs.get('block', None)
        self.optimize_history_save_path = general_config['optimize_history_save_path']
        self.general_hyperparameters = general_config['hyperparameters']
        self.label_transformer = kwargs.get('label_transformer')
        self.train_dataset, self.val_dataset = kwargs.get('dataset_tuple')
        self.loss_function = None
        self.metric_function = None
        self.is_cuda = None
        self.epoch = None
        self.eval_interval = None
        self.trial_times = None
        self.best_model_params = None
        self.network_info = None
        self.model_saver = None


    def initialize_train_parameters(self, **kwargs):
        self.loss_function = kwargs.get('loss_function')
        self.metric_function = kwargs.get('metric_function')
        self.is_cuda = kwargs.get('is_cuda', True)
        self.epoch = kwargs.get('epoch', 100)
        self.eval_interval = kwargs.get('eval_interval', 10)
        self.trial_times = kwargs.get('trial_times', 100)

    def train_once(self, model, optimizer, train_loader):
        model.train()
        images, labels = next(iter(train_loader))
        output = model(images)
        # 从归一化的值恢复到真实值
        labels = self.label_transformer.label2real(labels)
        output = self.label_transformer.label2real(output)
        loss = self.loss_function(output, labels)
        # 计算MPD
        mpd = self.metric_function(labels, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), mpd

    def eval_once(self, model, eval_loader):
        model.eval()
        images, labels = next(iter(eval_loader))
        with torch.no_grad():
            output = model(images)
            # 从归一化的值恢复到真实值
            labels = self.label_transformer.label2real(labels)
            output = self.label_transformer.label2real(output)
            loss = self.loss_function(output, labels)
            # 计算MPD
            mpd = self.metric_function(labels, output)
            return loss.item(), mpd

    def objective(self, trial):
        # 超参数搜索空间
        optimizer_name = trial.suggest_categorical('optimizer', self.general_hyperparameters['optimizer'])
        lr = trial.suggest_float('learning_rate', self.general_hyperparameters['lr'][0], self.general_hyperparameters['lr'][1])
        batch_size = trial.suggest_categorical('batch_size', self.general_hyperparameters['batch_size'])
        dropout_rate = trial.suggest_float('dropout_rate', self.general_hyperparameters['dropout_rate'][0], self.general_hyperparameters['dropout_rate'][1])
        # 网络结构搜索空间
        block_name = trial.suggest_categorical('block', self.specific_config['block_keys'])
        mlp_name = trial.suggest_categorical('mlp', self.specific_config['mlp_keys'])
        # 初始化数据集
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型和损失函数
        self.general_config["dropout_rate"] = dropout_rate
        model = NetworkFrame(self.general_config, self.specific_config, self.block, block_name=block_name, mlp_name=mlp_name)
        if self.is_cuda:
            model.cuda()

        optimizer = None
        # 初始化优化器
        if optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.0, 0.9)  # SGD 动量
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_name == 'Adam':
            beta1 = trial.suggest_float('beta1', 0.8, 0.999)  # Adam beta1
            beta2 = trial.suggest_float('beta2', 0.9, 0.9999)  # Adam beta2
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        elif optimizer_name == 'RMSprop':
            alpha = trial.suggest_float('alpha', 0.8, 0.999)  # RMSprop alpha
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)
        if optimizer is None:
            raise ValueError('Invalid optimizer name: {}'.format(optimizer_name))

        # 初始化 tqdm 进度条
        final_loss = 10000
        final_res = 10000
        epoch_bar = tqdm(range(self.epoch), desc="Training", unit="epoch")
        for epoch in epoch_bar:
            train_loss, train_mpd = self.train_once(model, optimizer, train_loader)
            epoch_bar.set_postfix({"Train Loss": train_loss, "Train MPD": train_mpd})
            if epoch % self.eval_interval == 0 and epoch > 0:
                eval_loss, eval_mpd = self.eval_once(model, val_loader)
                epoch_bar.write(f"Eval at Epoch {epoch}: Loss: {eval_loss}, MPD: {eval_mpd}")
                if eval_loss < final_loss:
                    final_loss = eval_loss
                    final_res = final_loss + 0.1*eval_mpd
                    trial.report(final_res, epoch)
                    # 判断是否需要裁剪
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
        return final_res

    def start_trial(self):
        # 创建一个支持裁剪的采样器和裁剪器
        sampler = TPESampler()
        pruner = MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')
        study.optimize(self.objective, n_trials=self.trial_times)

        # 输出最佳超参数
        print('Best trial:')
        trial = study.best_trial
        self.best_model_params = trial.params
        print(f'  Value: {trial.value}')
        print('  Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')

        manage_folder(self.optimize_history_save_path, self.specific_config['name'])
        name_folder = os.path.join(self.optimize_history_save_path, self.specific_config['name'])

        # 保存最佳模型参数
        best_model_params_save_path = os.path.join(str(name_folder), "best_model_params.json")
        with open(best_model_params_save_path, "w", encoding="utf-8") as f:
            json.dump(trial.params, f, indent=4, ensure_ascii=False)

        # 可视化
        # (1) 绘制优化历史
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(str(name_folder), "optimization_history.png"), scale=5)

        # (2) 分析超参数重要性
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(str(name_folder), "param_importance.png"), scale=5)

        # (3) 绘制切片图
        fig3 = plot_slice(study)
        fig3.write_image(os.path.join(str(name_folder), "slice_plot.png"), scale=5)

        print("所有可视化结果已保存为图片文件！")

    def train_from_best_params(self, **kwargs):
        """
        基于最佳超参数重新训练模型
        """
        if not self.best_model_params:
            raise ValueError("Best model parameters are not set. Please run optimization first.")

        # 从 best_model_params 中提取超参数
        optimizer_name = self.best_model_params['optimizer']
        lr = self.best_model_params['learning_rate']
        batch_size = self.best_model_params['batch_size']
        dropout_rate = self.best_model_params['dropout_rate']
        block_name = self.best_model_params['block']
        mlp_name = self.best_model_params['mlp']

        # 初始化数据集
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # 更新通用配置中的 dropout_rate
        self.general_config["dropout_rate"] = dropout_rate

        # 初始化模型
        model = NetworkFrame(
            self.general_config,
            self.specific_config,
            self.block,
            block_name=block_name,
            mlp_name=mlp_name
        )
        if self.is_cuda:
            model.cuda()

        # 初始化优化器
        optimizer = None
        if optimizer_name == 'SGD':
            momentum = self.best_model_params.get('momentum', 0.9)  # 如果没有 momentum，默认值为 0.9
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_name == 'Adam':
            beta1 = self.best_model_params.get('beta1', 0.9)  # 如果没有 beta1，默认值为 0.9
            beta2 = self.best_model_params.get('beta2', 0.999)  # 如果没有 beta2，默认值为 0.999
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        elif optimizer_name == 'RMSprop':
            alpha = self.best_model_params.get('alpha', 0.99)  # 如果没有 alpha，默认值为 0.99
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)
        if optimizer is None:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        # 记录网络信息
        init_info = {
            "name": self.specific_config['name'],
            "block_type": self.specific_config["block_type"],
            "im_size": self.general_config['im_size'],
            "in_channels": self.general_config['in_channels'],
            "train_params": get_parameter_number(model),
            "block": self.specific_config['block'][block_name],
            "mlp": self.specific_config['mlp'][mlp_name],
        }
        self.network_info = NetworkInfo(init_info)
        # 模型保存信息
        model_save_info = {
            "save_path": self.general_config['save_path'],
            "loss_threshold": self.general_config['loss_threshold'],
            "name": self.specific_config['name'],
        }
        self.model_saver = ModelSaver(model_save_info)

        # 初始化 tqdm 进度条
        final_loss = float('inf')
        epoch = kwargs.get('epoch', 100)
        epoch_bar = tqdm(range(epoch), desc="Training from Best Params", unit="epoch")
        # 记录训练开始时间
        start_time = time.time()
        for epoch in epoch_bar:
            # 训练一次
            train_loss, train_mpd = self.train_once(model, optimizer, train_loader)
            final_loss = train_loss
            self.network_info.train_log.add_entry(epoch, train_loss, train_mpd)
            epoch_bar.set_postfix({"Train Loss": train_loss, "Train MPD": train_mpd})
            # 验证模型
            if epoch % self.eval_interval == 0 and epoch > 0:
                eval_loss, eval_mpd = self.eval_once(model, val_loader)
                self.network_info.eval_log.add_entry(epoch, eval_loss, eval_mpd)
                epoch_bar.write(f"Eval at Epoch {epoch}: Loss: {eval_loss}, MPD: {eval_mpd}")
            self.model_saver.save_checkpoint(model, optimizer, epoch, train_loss)
        # 保存最后的模型
        self.model_saver.save_model(model, final_loss)
        # 记录训练结束时间
        end_time = time.time()
        training_time = end_time - start_time
        train_info = {
            "train_time": training_time,
            "total_epoch": epoch + 1,
            "train_batch_size": batch_size,

            "optimizer": optimizer_name,
            "lr": lr,

            "dropout_rate": dropout_rate,
            "loss_func": kwargs.get('loss_function', 'VmLoss'),
            "eval_metric": kwargs.get('eval_metric', 'MPD'),
        }
        self.network_info.set_train_info(train_info)
        self.network_info.save_network_info(kwargs.get('save_log_config'))