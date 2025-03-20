import torch

class LabelTransform:
    def __init__(self, train_params, is_cuda):
        pose_params = None
        if train_params['standard_pose'] == "标准正位":
            pose_params = train_params['standard_front']
        elif train_params['standard_pose'] == "标准侧位":
            pose_params = train_params['standard_side']
        if pose_params is not None:
            # 将初始姿态和噪声范围转换为 PyTorch 张量
            self.init_pose = torch.tensor(pose_params['init_pose'])
            self.trans_noise_range = torch.tensor(pose_params['translation_noise_range'])
            self.rota_noise_range = torch.tensor(pose_params['rotation_noise_range'])
            if is_cuda:
                self.init_pose = self.init_pose.cuda()
                self.trans_noise_range = self.trans_noise_range.cuda()
                self.rota_noise_range = self.rota_noise_range.cuda()

    def label2real(self, label):
        """
        将标签（噪声值）转换为真值（真实的旋转和平移参数）。
        :param label: 包含噪声的标签，形状为 [batch_size, 6] 或 [6]，
                      分别表示 rx_noise, ry_noise, rz_noise, tx_noise, ty_noise, tz_noise。
        :return: 真实的旋转和平移参数，形状与输入相同。
        """
        # 确保输入是 PyTorch 张量
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)

        # 解析标签
        rx_noise = label[..., 0]  # 提取 rx_noise
        ry_noise = label[..., 1]  # 提取 ry_noise
        rz_noise = label[..., 2]  # 提取 rz_noise
        tx_noise = label[..., 3]  # 提取 tx_noise
        ty_noise = label[..., 4]  # 提取 ty_noise
        tz_noise = label[..., 5]  # 提取 tz_noise

        # 计算真实的旋转参数
        rx_true = rx_noise * self.rota_noise_range[0]
        ry_true = ry_noise * self.rota_noise_range[1]
        rz_true = rz_noise * self.rota_noise_range[2]

        # 计算真实的平移参数
        tx_true = tx_noise * self.trans_noise_range[0]
        ty_true = ty_noise * self.trans_noise_range[1]
        tz_true = tz_noise * self.trans_noise_range[2]

        # 返回真值
        return torch.stack([rx_true, ry_true, rz_true, tx_true, ty_true, tz_true], dim=-1)