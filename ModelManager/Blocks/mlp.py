import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        # 定义全连接层
        self.fc = nn.Linear(input_dim, output_dim)  # 输入层到隐藏层
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc(x))
        return x