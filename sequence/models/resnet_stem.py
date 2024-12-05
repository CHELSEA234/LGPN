import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import copy

# from .LaPlacianMs import LaPlacianMs

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()

        ## the new stem archi.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1_0 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_0 = nn.BatchNorm2d(64)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.stem_merge = self._make_merge(192, 64, dropout=False)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        ## high-res branch.
        self.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        ## feature reverse-pyramid.
        p_dim = 64
        self.latlayer0 = nn.Conv2d( 256, p_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d( 512, p_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, p_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(2048, p_dim, kernel_size=1, stride=1, padding=0)

        ## down sampling branch. 
        self.output_branch = nn.Sequential(
                                            nn.Conv2d(p_dim, p_dim, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(p_dim, p_dim, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(p_dim, p_dim, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),
                                            # nn.AdaptiveAvgPool2d((1, 1)),
                                            )
        self.fc_2 = nn.Linear(p_dim, 1)

        ## merge high-res branch into the main branch.
        self.conv_red_0 = self._make_merge(64, 16, True, 0.1)
        self.conv_red_1 = self._make_merge(64, 32, True, 0.1)
        self.conv_red_2 = self._make_merge(64, 64, True, 0.1)

        self.conv_smooth_0 = self._make_merge(256+16, 256)
        self.conv_smooth_1 = self._make_merge(512+32, 512)
        self.conv_smooth_2 = self._make_merge(1024+64, 1024)
        self.conv_smooth_last = self._make_merge(2048+64, 2048)
        self.merger_function = merge_concat

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        new_component_list = [
                                self.conv1_0, self.conv1_1, 
                                self.conv_red_0, self.conv_red_1, 
                                self.conv_red_2, # self.conv_red_last, # self.conv_red_3,
                                self.conv_smooth_0, self.conv_smooth_1, 
                                self.conv_smooth_2, self.conv_smooth_last, # self.conv_smooth_3,
                                self.latlayer0, self.latlayer1, self.latlayer2, self.latlayer3, 
                                self.output_branch,
                                self.fc_2
                                ]
        for component in new_component_list:
            for m in component.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_merge(self, input_dim, output_dim, dropout=False, dropout_val=0.2):
        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, bias=False, groups=2))
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(p=dropout_val))
        return nn.Sequential(*layers)

    def _initialize_freq(self, ):
        '''take the pre-trained weights from the pre-trained ResNet.'''
        # self.conv1_freq = copy.deepcopy(self.conv1)
        # self.bn1_freq = copy.deepcopy(self.bn1)
        # self.layer1_freq = copy.deepcopy(self.layer1)
        # print("Finish the copy-initialization here.")
        self.conv2 = copy.deepcopy(self.conv1)
        self.bn2 = copy.deepcopy(self.bn1)
        self.conv3 = copy.deepcopy(self.conv1)
        self.bn3 = copy.deepcopy(self.bn1)
        self.conv4 = copy.deepcopy(self.conv1)
        self.bn4 = copy.deepcopy(self.bn1)
        # print("Finish the copy-initialization here.")

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return torch.nn.functional.upsample(x, size=(H,W), mode='bilinear') + y

    def _upsample(self, x, y):
        _,_,H,W = y.size()
        _,_,H1,W1 = x.size()
        if H1 == H and W1 == W:
            return x
        else:
            return torch.nn.functional.upsample(x, size=(H,W), mode='bilinear')

    def forward(self, x):

        x_scale_0 = self.conv1(x)
        x_scale_0 = self.bn1(x_scale_0)
        x_scale_1 = self.conv1_0(x)
        x_scale_1 = self.bn1_0(x_scale_1)
        x_scale_2 = self.conv1_1(x)
        x_scale_1 = self.bn1_1(x_scale_1)
        x_scale = torch.cat((x_scale_0, x_scale_1, x_scale_2), 1)
        x_scale = self.relu(x_scale)
        c1 = self.stem_merge(x_scale)

        c2 = self.layer1(c1)      
        c3 = self.layer2(c2)      
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        x_1 = self.latlayer0(c2)
        x_1 = self._upsample_add(x_1, c1)

        x_2 = self.latlayer1(c3)
        x_2 = self._upsample_add(x_2, x_1)

        x_3 = self.latlayer2(c4)
        x_3 = self._upsample_add(x_3, x_2)

        x_4 = self.latlayer3(c5)
        x_4 = self._upsample_add(x_4, x_3)
        x_4 = self.output_branch(x_4)
        
        x = self.conv_smooth_last(self.merger_function(x_4, c5))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    model._initialize_freq()
    # print("comes to this place.")
    # import sys;sys.exit(0)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

## Functions to merger the bidirectional outputs
# Concatenation function
def merge_concat(out1, out2):
    return torch.cat((out1, out2), 1)
# Summation function
def merge_sum(out1, out2):
    return torch.add(out1, out2)