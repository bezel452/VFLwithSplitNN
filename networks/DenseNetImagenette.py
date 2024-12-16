import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.dropout = nn.Dropout(p=self.drop_rate)
    
    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class _TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.trainsition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=plance, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
    def forward(self, x):
        return self.trainsition_layer(x)

class DenseNetBottom(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks = [6, 12 ,24, 16]):
        super(DenseNetBottom, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=3, places=init_channels)
        
        num_features = init_channels

        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2

        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2

        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2

        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[3] * growth_rate
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x
    
class DenseNetTop(nn.Module):
    def __init__(self, num_clients,num_classes=10):
        super(DenseNetTop, self).__init__()
        self.num = num_clients
        self.fc1 = nn.Linear(1024 * self.num, 512 * self.num)
        self.bn0 = nn.BatchNorm1d(1024 * self.num)
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