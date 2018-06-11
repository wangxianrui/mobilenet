import torch
import torch.nn as nn
import config


class MobileNet(nn.Module):
    def __init__(self, num_class=config.num_classes):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            self.conv_bn(3, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_class)

    def conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def conv_dw(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
