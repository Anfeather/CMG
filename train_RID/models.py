import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch.nn.functional as F
encoded_image_size = 1
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fc = nn.Linear(2048*encoded_image_size*encoded_image_size, 2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        # print(out.shape)
        out = torch.flatten(out, 1)
        # out = out.permute(0, 2, 3, 1)
        # out = out.reshape(out.shape[0],-1,out.shape[-1])
        
        # out = (out.sum(1))/(out.shape[0])

       
        # out = self.fc(out)
        return out


    def forward_restricition(self,x,layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(out.shape[0],-1,out.shape[-1])
        return out



def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    "resnet18": [resnet18, 512],
    "resnet34": [resnet34, 512],
    "resnet50": [resnet50, 2048],
    "resnet101": [resnet101, 2048],
}




class classification_ResNet(nn.Module):
    def __init__(self, arch="resnet18", out_dim=128, num_classes=1, **kwargs):
        super(classification_ResNet, self).__init__()
        m, fdim = model_dict[arch]
        self.encoder = m()
        self.classifier = nn.Linear(fdim, num_classes)
        self.head = nn.Sequential(
            nn.Linear(fdim, fdim), nn.ReLU(inplace=True), nn.Linear(fdim, out_dim)
        )

    def get_features(self, x):
        return F.normalize(self.head(self.encoder(x)), dim=-1)

    def forward(self, x):
        return self.classifier(self.encoder(x))




class RankNet(nn.Module):
    def __init__(self, num_features):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(num_features,
                                             512), nn.Dropout(0.2), nn.ReLU(),
                                   nn.Linear(512, 256), nn.Dropout(0.2),
                                   nn.ReLU(), nn.Linear(256, 128))
                                #    nn.Dropout(0.2), nn.ReLU(),
                                #    nn.Linear(128, 1))
        # self.output = nn.Sigmoid()
        self.output = nn.Softmax(dim=1)
        
    def forward(self, input1, input2, input3, input4):
        s1 = self.model(input1)
        s2 = self.model(input2)
        s3 = self.model(input3)
        s4 = self.model(input4)

        # s_dot1 = torch.mm(s1,s2.t()) + torch.mm(s1,s3.t()) + torch.mm(s1,s4.t())
        # s_dot2 = torch.mm(s2,s1.t()) + torch.mm(s2,s3.t()) + torch.mm(s2,s4.t())
        # s_dot3 = torch.mm(s3,s1.t()) + torch.mm(s3,s2.t()) + torch.mm(s3,s4.t())
        # s_dot4 = torch.mm(s4,s1.t()) + torch.mm(s4,s2.t()) + torch.mm(s4,s3.t())
        s_dot1 = torch.sum( s1*s2 + s1*s3 + s1*s4, dim = 1,keepdim=True )
        s_dot2 = torch.sum( s2*s1 + s2*s3 + s2*s4, dim = 1,keepdim=True )
        s_dot3 = torch.sum( s3*s1 + s3*s2 + s3*s4, dim = 1,keepdim=True )
        s_dot4 = torch.sum( s4*s1 + s4*s2 + s4*s3, dim = 1,keepdim=True )
        
        sum_dot = s_dot1+s_dot2+s_dot3+s_dot4
        s_dot1 = s_dot1/sum_dot
        s_dot2 = s_dot2/sum_dot
        s_dot3 = s_dot3/sum_dot
        s_dot4 = s_dot4/sum_dot
        s = torch.cat([s_dot1,s_dot2,s_dot3,s_dot4],dim=1)
        prob = self.output(s)

        return prob

    def predict(self, input_):
        return self.forward(input_)