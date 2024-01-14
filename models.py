"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
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





class SSLResNet(nn.Module):
    def __init__(self, arch="resnet50", out_dim=10, **kwargs):
        super(SSLResNet, self).__init__()
        m, fdim = model_dict[arch]
        self.encoder = m()
        self.head = nn.Sequential(
            nn.Linear(fdim, fdim), nn.ReLU(inplace=True), nn.Linear(fdim, out_dim)
        )

    def forward(self, x):
        
        return F.normalize(self.head(self.encoder(x)), dim=-1)


class TextNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: dimension of tags
        :param output_dim: bit number of the final binary code
        """
        super(TextNet, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        mid_num = 4096
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        x = x / norm
        return x




class Encoder50(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14, class_num = 30):
        super(Encoder50, self).__init__()
        self.enc_image_size = encoded_image_size

        self.resnet = torchvision.models.resnet50(pretrained=False)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers
        # modules = list(resnet.children())[:-2]
        # self.resnet = nn.Sequential(*modules)

        self.img_feature = nn.Sequential(*list(self.resnet.children())[:-2])

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
          
        channel_in = self.resnet.fc.in_features  
        self.resnet.fc = nn.Linear(channel_in, class_num)  

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)

        return out

    def encoder(self, images):

        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # print(out.size)
        return out


class VGG(nn.Module):
    """
    VGG builder
    """
    def __init__(self, arch: object, num_classes=1000) -> object:
        super(VGG, self).__init__()
        self.in_channels = 3
        self.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(25088, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 2048)
        # self.fc3 = nn.Linear(4096, num_classes)
        self.head = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(inplace=True), nn.Linear(2048, 128)
        )


    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def encoder(self, x):

        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
  
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

    def forward(self, x):
        # # print("X:",x.shape)
        # # out = self.conv3_64(x)
        # # out = F.max_pool2d(out, 2)
        # # out = self.conv3_128(out)
        # # out = F.max_pool2d(out, 2)
        # # out = self.conv3_256(out)
        # # out = F.max_pool2d(out, 2)
        # # out = self.conv3_512a(out)
        # # out = F.max_pool2d(out, 2)
        # # out = self.conv3_512b(out)
        # # # print(out.shape)
        # # out = F.max_pool2d(out, 2)
        # out = x.view(x.size(0), -1)
        # out = self.fc1(out)
        # out = self.bn1(out)
        # out = F.relu(out)
        # out = self.fc2(out)
        # # out = self.bn2(out)
        # # out = F.relu(out)
        return F.normalize(self.head(self.encoder(x)), dim=-1)




def VGG_11():
    return VGG([1, 1, 2, 2, 2], num_classes=30)

def VGG_13():
    return VGG([1, 1, 2, 2, 2], num_classes=30)

def VGG_16():
    return VGG([2, 2, 3, 3, 3], num_classes=30)
def VGG_19():
    return VGG([2, 2, 4, 4, 4], num_classes=30)
import torch
import torch.nn as nn
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, num_classes=10):
        super(DenseNet_BC, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        # 表示cifar-10
        if num_classes == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))



        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.head = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(inplace=True), nn.Linear(2048, 128)
        )

        self.classifier = nn.Linear(num_feature, 2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def encoder(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
    def forward(self, x):
        return F.normalize(self.head(self.encoder(x)), dim=-1)
        
# DenseNet_BC for ImageNet
def DenseNet121():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=30)

def DenseNet169():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=30)

def DenseNet201():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=30)

def DenseNet161():
    return DenseNet_BC(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=30)

# DenseNet_BC for cifar
def densenet_BC_100():
    return DenseNet_BC(growth_rate=12, block_config=(16, 16, 16))



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


class classification_TextNet(nn.Module):
    def __init__(self, input_dim, output_dim,num_classes=30):
        """
        :param input_dim: dimension of tags
        :param output_dim: bit number of the final binary code
        """
        super(classification_TextNet, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        mid_num = 4096
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, output_dim)
        self.classifier = nn.Linear(output_dim, num_classes)
    def get_features(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        x = x / norm
        return x

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        x = x / norm
        return self.classifier(x)
        


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