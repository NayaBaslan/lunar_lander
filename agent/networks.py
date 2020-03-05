import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision

"""
Imitation learning network
"""


class FCN(nn.Module):

    def __init__(self, hidden_layers, history_length, dim_state=8, n_classes=4):
        super(FCN, self).__init__()
        # TODO : define layers of a  fully-connected neural network

        self.fc1 = nn.Linear(dim_state, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc8 = nn.Linear(hidden_layers, n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc8(x)
        return x


class CNN(nn.Module):

    def __init__(self, history_length, n_classes=4):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.conv1 = nn.Conv2d(history_length, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(17280, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, 17280)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class baseBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(baseBlock, self).__init__()
        # declare convolutional layers with batch norms
        self.conv1 = torch.nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        # Save the residue
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output


class ResNet(torch.nn.Module):

    def __init__(self, history_length, num_layers, block=baseBlock, classes=4):
        super(ResNet, self).__init__()
        # according to research paper:
        self.input_planes = 64
        self.conv1 = torch.nn.Conv2d(history_length, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer2 = self._layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 512, num_layers[3], stride=2)
        self.fc_1 = torch.nn.Linear(126464, 4048)
        self.fc_2 = torch.nn.Linear(4048, 1024)
        self.fc_out = torch.nn.Linear(1024, classes)

    def _layer(self, block, planes, num_layers, stride=1):

        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
        return torch.nn.Sequential(*netLayers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_out(x)

        return x
