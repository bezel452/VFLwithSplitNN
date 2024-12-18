import torch
import torch.nn as nn

class LogisticRegressionBottom(nn.Module):
    def __init__(self, num_input, num_output):
        super(LogisticRegressionBottom, self).__init__()
        self.linear = nn.Linear(num_input, num_output)
        self.sigmoid = nn.Softmax()

    def forward(self, x):
        x = self.linear(x)
        return x

class LRTop(nn.Module):
    def __init__(self, num_clients, num_classes = 2):
        super(LRTop, self).__init__()
        self.num = num_clients
        self.linear = nn.Linear(2 * num_clients, num_classes)
        self.sigmoid = nn.Softmax()

    def forward(self, inputs):
        x = inputs[0]
        for i in range(1, self.num):
            x = torch.cat((x, inputs[i]), dim=1)
        x = self.sigmoid(self.linear(x))
        return x