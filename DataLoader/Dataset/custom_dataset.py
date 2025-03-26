import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, Subset
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


def split_dataset(
        dataset,
        split_history_dir="data_splits",
        split_history_filename="default_split.json",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        overwrite=False
):
    """
    按比例划分数据集（支持从历史文件加载）

    :param dataset: 完整数据集
    :param split_history_dir: 划分记录保存目录
    :param split_history_filename: 划分记录文件名
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param random_seed: 随机种子
    :param overwrite: 是否强制重新划分（覆盖旧记录）
    :return: (train_dataset, val_dataset, test_dataset)
    """
    os.makedirs(split_history_dir, exist_ok=True)
    split_path = os.path.join(split_history_dir, split_history_filename)

    # 如果存在记录文件且不强制覆盖，则加载历史划分
    if not overwrite and os.path.exists(split_path):
        with open(split_path, 'r') as f:
            indices = json.load(f)

        # 验证数据集大小是否匹配
        if len(indices['train']) + len(indices['val']) + len(indices['test']) == len(dataset):
            return (
                Subset(dataset, indices['train']),
                Subset(dataset, indices['val']),
                Subset(dataset, indices['test'])
            )
        print("Warning: Dataset size mismatch, regenerating splits...")

    # 执行新划分
    assert abs((train_ratio + val_ratio + test_ratio) - 1) < 1e-6, "比例之和必须为1"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # 保存划分记录
    indices = {
        'train': train_dataset.indices,
        'val': val_dataset.indices,
        'test': test_dataset.indices,
        'metadata': {
            'total_size': total_size,
            'ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
            'seed': random_seed
        }
    }

    with open(split_path, 'w') as f:
        json.dump(indices, f, indent=2)

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