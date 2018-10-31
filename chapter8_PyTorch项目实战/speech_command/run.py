from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from speech_loader import SpeechLoader
import numpy as np
from model import VGG
from train import train, test
import torch.nn.functional as F
from torch.autograd import Variable

# 参数设置
parser = argparse.ArgumentParser(description='Google Speech Commands Recognition')
parser.add_argument('--train_path', default='data/train', help='path to the train data folder')
parser.add_argument('--test_path', default='data/test', help='path to the test data folder')
parser.add_argument('--valid_path', default='data/valid', help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='VGG11', help='network architecture: VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')

# 特征提取参数设置
parser.add_argument('--window_size', default=.02, help='window size for the stft')
parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
parser.add_argument('--window_type', default='hamming', help='window type for the stft')
parser.add_argument('--normalize', default=True, help='boolean, wheather or not to normalize the spect')

args = parser.parse_args()

# 确定是否使用CUDA
args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)  # PyTorch随机种子设置
if args.cuda:
    torch.cuda.manual_seed(args.seed) # CUDA随机种子设置

# 加载数据, 训练集,验证集和测试集
train_dataset = SpeechLoader(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type, normalize=args.normalize)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=20, pin_memory=args.cuda, sampler=None)

valid_dataset = SpeechLoader(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type, normalize=args.normalize)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

test_dataset = SpeechLoader(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
                              window_type=args.window_type, normalize=args.normalize)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

# 建立网络模型
model = VGG(args.arc)

if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

# 定义优化器
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#import pdb
#pdb.set_trace()
# train 和　valid过程
for epoch in range(1, args.epochs + 1):
    # 模型在train集上训练
    train(train_loader, model, optimizer, epoch, args.cuda)

    # 验证集测试
    test(valid_loader, model, args.cuda, 'valid')

# 测试集验证
test(test_loader, model, args.cuda, 'test')


