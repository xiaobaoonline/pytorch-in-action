# coding=utf-8
# 配置库
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets


# load model
# 保存模型
# torch.save(model.state_dict(), './cnn.pth')

# 定义卷积神经网络模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):  # 28x28x1
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),  # 28 x28
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14 x 14
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # 10 * 10*16
            nn.ReLU(True), nn.MaxPool2d(2, 2))  # 5x5x16

        self.fc = nn.Sequential(
            nn.Linear(400, 120),  # 400 = 5 * 5 * 16
            nn.Linear(120, 84),
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 400)  # 400 = 5 * 5 * 16, 
        out = self.fc(out)
        return out


# 打印模型
print(Cnn)

model = Cnn(1, 10)  # 图片大小是28x28, 10
# cnn = torch.load('./cnn.pth')['state_dict']
model.load_state_dict(torch.load('./cnn.pth'))

# 识别
print(model)
test_data = datasets.MNIST(root='./data', train=False, download=True)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[
         :20] / 255.0  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:20]
print(test_x.size())
test_output = model(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'predict result')
print(test_y[:10].numpy(), 'real result')
