import os
import pdb
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt

# 配置参数
torch.manual_seed(1) #设置随机数种子，确保结果可重复                                                                                                      
batch_size = 128  #批处理大小
learning_rate = 1e-2  #学习率
num_epochs = 10      #训练次数   

#下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
        root='./data',  #数据保持的位置
        train=True, # 训练集 
        transform=transforms.ToTensor(),# 一个取值范围是[0,255]的PIL.Image
                        # 转化为取值范围是[0,1.0]的torch.FloadTensor
        download=True) #下载数据

test_dataset = datasets.MNIST(
       root='./data', 
       train=False, # 测试集
       transform=transforms.ToTensor())

#pdb.set_trace()
#数据的批处理，尺寸大小为batch_size, 
#在训练集中，shuffle 必须设置为True, 表示次序是随机的
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 500),
            nn.ReLU(True), 
            nn.Linear(500, 250), 
            nn.ReLU(True), 
            nn.Linear(250, 2)
            )
        self.decoder = nn.Sequential(
            nn.Linear(2, 250),
            nn.ReLU(True),
            nn.Linear(250, 500),
            nn.ReLU(True),
            nn.Linear(500, 1000),
            nn.ReLU(True), 
            nn.Linear(1000, 28 * 28), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#model = autoencoder().cuda()
model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        #img = Variable(img).cuda()
        img = Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))


#模型测试， 由于训练和测试 BatchNorm, Dropout配置不同，需要说明是否模型测试
model.eval()
eval_loss = 0
import pdb
#pdb.set_trace()
for data in test_loader:  #test set 批处理
    img, label = data

    img = img.view(img.size(0), -1)
    #img = Variable(img, volatile=True).cuda() # volatile 确定你是否不调用.backward(), 测试中不需要
    img = Variable(img, volatile=True) 
    label = Variable(label, volatile=True)
    out = model(img)  # 前向算法 
    out = out.detach().numpy()
    y = (label.data).numpy()
    plt.scatter(out[:, 0], out[:, 1], c = y)
    plt.colorbar()
    plt.title('audocoder of MNIST test dataset')
    plt.show()


