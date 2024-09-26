'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, bn_layer, stride=1, bn_partition=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if bn_partition == 1:
            self.bn1 = bn_layer(planes)
            self.bn2 = bn_layer(planes)
        else:
            self.bn1 = bn_layer(planes,bn_partition)
            self.bn2 = bn_layer(planes,bn_partition)
            
        #self.bn1 = BatchNorm2dPartition(planes,bn_partition)
        #self.bn1 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        if stride != 1 or in_planes != self.expansion*planes:
            if bn_partition == 1:
                self.shortcut = nn.Sequential(
                    conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    bn_layer(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    bn_layer(self.expansion*planes, bn_partition)
                )
                
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        out = self.relu(out)
        
        #out_cpu = out.cpu().detach().numpy()
        #print(out_cpu.shape)
        #for i in range(out_cpu.shape[1]):
        #    print(i, np.sum(out_cpu[:,i,:,:]))
        
        
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_layer, stride=1, bn_partition=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2dPartition(planes,bn_partition)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2dPartition(planes,bn_partition)
        self.conv3 = conv_layer(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2dPartition(self.expansion*planes,bn_partition)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2dPartition(self.expansion*planes,num_bn)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, conv_layer, bn_layer, num_classes=10, num_filters=512, bn_partition=[1]*9):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.shrink = num_filters/512
        self.bn_partition = bn_partition
        
        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # num_bn = self.bn_partition.pop(0)
        num_bn = self.bn_partition
        self.bn1 = bn_layer(64) if num_bn==1 else bn_layer(64, num_bn)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64*self.shrink),  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*self.shrink), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*self.shrink), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_filters, num_blocks[3], stride=2)

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(4)

        # self.linear1 = nn.Linear(num_filters*block.expansion, 256, bias=False)
        # self.linear2 = nn.Linear(256, 128, bias=False)
        # self.out = nn.Linear(128, num_classes)


        self.out= nn.Linear(num_filters*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # layers.append(block(self.in_planes, planes, self.conv_layer, self.bn_layer, stride, self.bn_partition.pop(0)))
            layers.append(block(self.in_planes, planes, self.conv_layer, self.bn_layer, stride, self.bn_partition))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
            
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        
        #out_cpu = out.cpu().detach().numpy()
        #print(out_cpu.shape)
        #for i in range(out_cpu.shape[1]):
        #    print(i, np.sum(out_cpu[:,i]))
        # 
        # np.sum(weight_copy[i,:,:,:])




        # out = self.relu(self.linear1(out))
        # out = self.relu(self.linear2(out))
        # out = self.out(out)
        # return out

        out = self.out(out)
        #np.sum(weight_copy[i,:,:,:])
        return out

def resnet18(conv_layer, bn_layer, **kwargs):
    # bn_partition = kwargs['bn_partition'] if 'bn_partition' in kwargs else [1]*9
    bn_partition = 1
    return ResNet(BasicBlock, [2,2,2,2], conv_layer, bn_layer, num_classes=kwargs['num_classes'], bn_partition=bn_partition)

def resnet34(conv_layer, bn_layer, **kwargs):
    # bn_partition = kwargs['bn_partition'] if 'bn_partition' in kwargs else [1]*9
    bn_partition = 1
    return ResNet(BasicBlock, [3,4,6,3], conv_layer, bn_layer, num_classes=kwargs['num_classes'], bn_partition=bn_partition)

def resnet50(conv_layer, bn_layer, **kwargs):
    # bn_partition = kwargs['bn_partition'] if 'bn_partition' in kwargs else [1]*9
    bn_partition = 1
    return ResNet(BasicBlock, [3,4,6,3], conv_layer, bn_layer, num_classes=kwargs['num_classes'], bn_partition=bn_partition)

def resnet101(conv_layer, bn_layer, **kwargs):
    # bn_partition = kwargs['bn_partition'] if 'bn_partition' in kwargs else [1]*9
    bn_partition = 1
    return ResNet(BasicBlock, [3,4,23,3], conv_layer, bn_layer, num_classes=kwargs['num_classes'], bn_partition=bn_partition)

def resnet152(conv_layer, bn_layer, **kwargs):
    # bn_partition = kwargs['bn_partition'] if 'bn_partition' in kwargs else [1]*9
    bn_partition = 1
    return ResNet(BasicBlock, [3,8,36,3], conv_layer, bn_layer, num_classes=kwargs['num_classes'], bn_partition=bn_partition)

# def resnet34(**kwargs):
#     rob = kwargs['robustness'] if 'robustness' in kwargs else False
#     return ResNet(BasicBlock, [3,4,6,3], kwargs['num_classes'], rob=rob)

# def resnet50(**kwargs):
#     rob = kwargs['robustness'] if 'robustness' in kwargs else False
#     return ResNet(Bottleneck, [3,4,6,3], kwargs['num_classes'], rob=rob)
    

# def resnet101(**kwargs):
#     rob = kwargs['robustness'] if 'robustness' in kwargs else False
#     return ResNet(Bottleneck, [3,4,23,3], kwargs['num_classes'], rob=rob)


# def resnet152(**kwargs):
#     rob = kwargs['robustness'] if 'robustness' in kwargs else False
#     return ResNet(Bottleneck, [3,8,36,3], kwargs['num_classes'], rob=rob)



# def test():
if __name__ == '__main__':
    net = resnet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())