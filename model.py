import torch
import torch.nn as nn
import numpy as np
class Bottleneck(nn.Module):  #Basic模块
    def __init__(self, inputs, outputs, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.conv3 = nn.Conv2d(outputs, outputs * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outputs * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(x)
        return x

class BasicBlock(nn.Module): #Basic模块
    expansion = 1
    def __init__(self, inputs, outputs, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class StageModule(nn.Module): #构建Stage
    def __init__(self, inputs, outputs, first_channel):
        super(StageModule, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.branches = nn.ModuleList()
        for i in range(self.inputs):  # 首先通过四个BasicBlock模块
            w = first_channel * (2 ** i)  # 对应分支的通道数，2的倍数增加
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)
        self.fuse_layers = nn.ModuleList()
        for i in range(self.outputs):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.inputs):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:   #输出分支小于输入分支时，需要对输入分支进行上采样
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1),
                            nn.BatchNorm2d(c * (2 ** i)),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else: #输出分支小于输入分支时，需要对输入分支进行下采样
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1),
                                nn.BatchNorm2d(c * (2 ** j)),
                                nn.ReLU(inplace=True)
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(c * (2 ** i))
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        output = []
        for i in range(len(self.fuse_layers)):
            output.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )
        return output

class HRnet(nn.Module):
    def __init__(self, first_channel = 32, num_joints = 27):
        super(HRnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 第一阶段
        downsample = nn.Sequential(  #通道需要变成四倍
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, first_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(first_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, first_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(first_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        # 第二阶段
        self.stage2 = nn.Sequential(
            StageModule(2, 2, first_channel)
        )
        self.transition2 = nn.ModuleList([
            nn.Identity(),  
            nn.Identity(), 
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(first_channel * 2, first_channel * 4, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(first_channel * 4),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        # 第三阶段
        self.stage3 = nn.Sequential(
            StageModule(3, 3, first_channel),
            StageModule(3, 3, first_channel),
            StageModule(3, 3, first_channel),
            StageModule(3, 3, first_channel)
        )

        self.transition3 = nn.ModuleList([
            nn.Identity(), 
            nn.Identity(),  
            nn.Identity(), 
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(first_channel * 4, first_channel * 8, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(first_channel * 8),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        # 第四阶段
        self.stage4 = nn.Sequential(
            StageModule(4, 4, first_channel),
            StageModule(4, 4, first_channel),
            StageModule(4, 1, first_channel)
        )
        self.out = nn.Conv2d(first_channel, num_joints, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1] 
        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  
        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  
        x = self.stage4(x)
        x = self.out(x[0])
        return x