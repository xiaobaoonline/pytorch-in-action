from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable

# 训练函数， 模型在train集上训练
def train(loader, model, optimizer, epoch, cuda):
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss = train_loss / len(loader.dataset)
    print('train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, train_correct, len(loader.dataset), 100. * train_correct / len(loader.dataset)))

# 测试函数,用来测试valid集和test集
def test(loader, model, cuda, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)

    print(dataset + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
