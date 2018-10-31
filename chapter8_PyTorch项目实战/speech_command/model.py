import torch.nn as nn
import torch.nn.functional as F

# 建立VGG卷积神经的模型层
def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':  # maxpool 池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # 卷积层
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)] #avgPool 池化层
    return nn.Sequential(*layers)

# 各个VGG模型的参数
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# VGG卷积神经网络
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])   # VGG的模型层
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 30)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # flatting
        out = self.fc1(out)  # 线性层
        out = self.fc2(out)  # 线性层
        #import pdb
        #pdb.set_trace()
        return F.log_softmax(out, dim=1)  # log_softmax 激活函数
