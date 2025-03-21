"""
负责保存模型
"""
import os

import torch

from utils.dire_check import manage_folder


class ModelSaver:
    def __init__(self, model_save_config: dict):
        self.save_path = model_save_config["save_path"]
        self.name = model_save_config["name"]
        self.loss_threshold = model_save_config["loss_threshold"]
        # 创建保存文件夹
        manage_folder(self.save_path, self.name)
        self.name_folder = os.path.join(self.save_path, self.name)
        # 创建checkpoints文件夹
        manage_folder(self.name_folder, "checkpoints")
        self.checkpoint_save_dir = os.path.join(str(self.name_folder), "checkpoints")

    def save_checkpoint(self, train_model, optimizer, epoch, loss):
        if loss > self.loss_threshold:
            return
        self.loss_threshold = loss
        # 保存断点
        checkpoint = {"model_state_dict": train_model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}

        # 格式化 loss 为小数点后 5 位
        formatted_loss = f"{loss:.5f}"
        torch.save(checkpoint, os.path.join(self.checkpoint_save_dir, f"checkpoint_ep{epoch}_loss{formatted_loss}.pth"))

    def save_model(self, train_model, loss):
        # 格式化 loss 为小数点后 5 位
        formatted_loss = f"{loss:.5f}"
        torch.save(train_model.state_dict(), os.path.join(str(self.name_folder), f"{self.name}_loss{formatted_loss}.pth"))