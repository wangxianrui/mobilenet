import torch
import torch.nn as nn
import torch.nn.functional
import config


class LinearBottleneck(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)

        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)

        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNetV2(nn.Module):

    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.activation = nn.ReLU6(inplace=True)

        self.t = [0, 1, 6, 6, 6, 6, 6, 6]
        self.c = [32, 16, 24, 32, 64, 96, 160, 320]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]

        self.conv1 = nn.Conv2d(3, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        self.conv_last = nn.Conv2d(self.c[-1], 1280, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(1280)

        self.avgpool = nn.AvgPool2d(7)
        # self.dropout = nn.Dropout(p=0.2, inplace=True)  # confirmed by paper authors
        self.fc = nn.Linear(1280, config.num_classes)

    def _make_repeat(self, inchannel, outchannel, t, n, s):
        repeat = []

        # first bottle
        repeat.append(LinearBottleneck(inplanes=inchannel, outplanes=outchannel, stride=s, t=t))

        if n > 1:
            for i in range(n - 1):
                repeat.append(LinearBottleneck(inplanes=outchannel, outplanes=outchannel, stride=1, t=t))
        return nn.Sequential(*repeat)

    def _make_bottlenecks(self):
        bottlenecks = []

        for i in range(len(self.c) - 1):
            bottlenecks.append(self._make_repeat(self.c[i], self.c[i + 1], self.t[i + 1], self.n[i + 1], self.s[i + 1]))

        return nn.Sequential(*bottlenecks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)

        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        x = self.avgpool(x)
        # x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=1)
