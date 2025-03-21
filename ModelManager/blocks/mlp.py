import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        # 定义全连接层
        self.fc = nn.Linear(input_dim, output_dim)  # 输入层到隐藏层
        self.dropout = nn.Dropout(dropout_rate)  # 隐藏层到输出层
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        x = self.tanh(x)
        return x