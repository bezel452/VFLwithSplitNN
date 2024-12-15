import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) \
            if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False)
        ])

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)

class ResNet50Bottom(nn.Module):
    def __init__(self, num_classes = 100):
        super(ResNet50Bottom, self).__init__()
        self.stem = nn.Sequential(*[
            Conv(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.stages = nn.Sequential(*[
            self._make_stage(64, 256, down_sample=False, num_blocks=3),
            self._make_stage(256, 512, down_sample=True, num_blocks=4),
            self._make_stage(512, 1024, down_sample=True, num_blocks=6),
            self._make_stage(1024, 2048, down_sample=True, num_blocks=3),
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, num_blocks):
        layers = [Bottleneck(in_channels, out_channels, down_sample=down_sample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stages(self.stem(x))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

class ResNet50Top(nn.Module):
    def __init__(self, num_clients, num_classes=100):
        super(ResNet50Top, self).__init__()
        self.num = num_clients
        self.fc1 = nn.Linear(2048 * self.num, 512 * self.num)
        self.bn0 = nn.BatchNorm1d(2048 * self.num)
        self.fc2 = nn.Linear(512 * self.num, 512)
        self.bn1 = nn.BatchNorm1d(512 * self.num)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, inputs):
        x = inputs[0]
        for i in range(1, self.num):
            x = torch.cat((x, inputs[i]), dim=1)
        x = self.fc1(F.relu(self.bn0(x)))
        x = self.fc2(F.relu(self.bn1(x)))
        x = self.fc3(F.relu(self.bn2(x)))
        return F.log_softmax(x, dim=1)