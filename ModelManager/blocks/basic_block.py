import torch.nn as nn

class BasicBlock(nn.Module):
    # 构造（3 * 3卷积 + 2 * 2 池化）
    def __init__(self, in_channel, out_channel, dropout_rate=0.5, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.maxpool(out)
        out = self.tanh(out)
        return out