import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from .image_process import norm_image, generate_uv_coordinates, expand_uv_to_batch


def string_to_list(input_str):
    """
    将字符串转换为列表的通用函数。
    如果转换失败，打印传入的参数并返回 None。
    """
    try:
        # 尝试将字符串转换为列表
        result = list(eval(input_str))
        return result
    except Exception as e:
        # 如果转换失败，打印传入的参数
        print(f"无法将以下参数转换为列表: {input_str}")
        return None

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    按比例划分数据集
    :param dataset: 完整的数据集
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param random_seed: 随机种子
    :return: 训练集、验证集、测试集
    """
    assert train_ratio + val_ratio + test_ratio == 1, "比例之和必须为 1"

    # 计算各数据集的大小
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed)
    )

    return train_dataset, val_dataset, test_dataset

class CustomDataset(Dataset):
    def __init__(self, config):
        """
        初始化数据集
        :param config: 配置文件
        """
        self.data_dir = config['data_path']
        self.is_cuda = config['is_cuda']
        self.images_dir = os.path.join(self.data_dir, "images")
        self.labels_path = os.path.join(self.data_dir, "label.json")
        self.config = config
        if self.config["is_add_uv_coord"]:
            self.uv = generate_uv_coordinates(config['im_sz'])

        # 加载标签文件
        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)

        # 提取元数据
        self.init_pose = self.labels.pop("init_pose", None)  # 提取并移除 "init_pose"
        self.total_num = self.labels.pop("total_num", None)  # 提取并移除 "total_num"
        self.rota_noise_range = string_to_list(self.labels.pop("rota_noise_range", None))  # 提取并移除 "rota_noise_range"
        self.trans_noise_range = string_to_list(self.labels.pop("trans_noise_range", None))  # 提取并移除 "trans_noise_range"

        # 获取所有数据文件名
        self.file_names = list(self.labels.keys())

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 样本索引
        :return: 图像数据和对应的标签
        """
        file_name = self.file_names[idx]
        file_path = os.path.join(self.images_dir, file_name)

        # 加载 .npy 文件
        image = np.load(file_path)
        if self.config["is_add_uv_coord"]:
            image = np.concatenate((image, self.uv), axis=0)

        # 生成标签
        label = np.array([float(self.labels[file_name]["rx_noise"])/ self.rota_noise_range[0],
                          float(self.labels[file_name]["ry_noise"])/ self.rota_noise_range[1],
                          float(self.labels[file_name]["rz_noise"])/ self.rota_noise_range[2],
                          float(self.labels[file_name]["tx_noise"])/ self.trans_noise_range[0],
                          float(self.labels[file_name]["ty_noise"])/ self.trans_noise_range[1],
                          float(self.labels[file_name]["tz_noise"])/ self.trans_noise_range[2]])

        image = norm_image(image.astype(np.float64))  / 255 # 归一化图像数据
        if self.is_cuda:
            image = torch.from_numpy(image).float().cuda()
            label = torch.from_numpy(label).float().cuda()
        return image, label