import torch
from torch import nn
from .Blocks import BasicBlock, MLP
from torchviz import make_dot

from .network_info import NetworkInfo


# 用于统计网络的可训练参数个数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class NetworkFrame(nn.Module):
    def __init__(self, general_config: dict, specific_config: dict, block):
        im_size = general_config['im_size']
        in_channels = general_config['in_channels']
        block_channels = specific_config['block_channels']
        mlp_channels = specific_config['mlp_channels']
        out_channels = general_config['out_channels']

        super(NetworkFrame, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=block_channels[0], kernel_size=(3, 3))
        # 特征提取层
        self.feature_layer = self._make_layer(block_channels, block)
        # 动态计算卷积层的输出维度
        self.conv_output_dim = self._calculate_conv_output_dim(im_size, in_channels)
        # 全连接层
        mlp_channels = [self.conv_output_dim] + mlp_channels + [out_channels]
        self.fc = self._make_layer(mlp_channels, MLP)
        # 是否保存网络结构图
        is_save_network_structure = specific_config.get('is_save_network_structure', False)
        if is_save_network_structure:
            self.get_network_structure(im_size, in_channels, specific_config['save_path'])
        # 记录网络信息
        init_info = {
            "name": specific_config['name'],
            "im_size": im_size,
            "in_channels": in_channels,
            "train_params": get_parameter_number(self)
        }
        self.network_info = NetworkInfo(init_info)


    @staticmethod
    def _make_layer(channels, block):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(block(channels[i], channels[i+1]))
        return nn.Sequential(*layers)

    def _calculate_conv_output_dim(self, im_size, in_channels):
        """
        通过一个前向传播的“探测”过程计算卷积层的输出维度
        """
        # 创建一个随机的输入张量（假设输入图像大小为 64x64）
        x = torch.randn(1, in_channels, im_size[0], im_size[1])  # [batch_size, channels, height, width]

        # 前向传播通过卷积层和池化层
        x = self.conv0(x)
        x = self.feature_layer(x)

        # 计算展平后的维度
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def test(self, im_size, in_channels):
        x = torch.randn(1, in_channels, im_size[0], im_size[1])
        y = self.forward(x)
        print(y.shape)

    def get_network_structure(self,im_size, in_channels, save_path):
        """
        获取网络结构
        """
        x = torch.randn(1, in_channels, im_size[0], im_size[1]).requires_grad_(True)
        y = self.forward(x)
        MyConvNetVis = make_dot(y, params=dict(list(self.named_parameters()) + [('x', x)]))
        MyConvNetVis.format = "png"
        MyConvNetVis.directory = save_path
        MyConvNetVis.view()