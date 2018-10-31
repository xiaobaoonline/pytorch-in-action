import argparse
import random

import model
import mydatasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data

# import torch.nn.init.xavier_uniform_ as xavier
# random_state = 11892  #92%
random_state = 11117  # 20  94.22%
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
random.seed(random_state)

# lr = 0.001 , 17 ,95.11%
# 13 , 96.88%

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 20]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout [default: 0.5]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=128, help='number of each kind of kernel, 100')
parser.add_argument('-kernel-sizes', type=str, default='3,5,7',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

args = parser.parse_args()


# load 20new dataset
def new_20(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.NEWS_20.splits(text_field, label_field)

    max_document_length = max([len(x.text) for x in train_data.examples])
    print('train max_document_length', max_document_length)

    max_document_length = max([len(x.text) for x in dev_data])
    print('dev max_document_length', max_document_length)

    text_field.build_vocab(train_data, dev_data)
    text_field.vocab.load_vectors('glove.6B.100d')

    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(args.batch_size, len(dev_data)),
        **kargs)
    return train_iter, dev_iter, text_field


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, text_field = new_20(text_field, label_field, device=-1, repeat=False)

# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1

args.cuda = (not args.no_cuda) and torch.cuda.is_available();
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)

# load pre-training  glove model
cnn.embed.weight.data = text_field.vocab.vectors
#  weight init
cnn.apply(weights_init)  #

# print net
print(cnn)
'''

CNN_Text(
  (embed): Embedding(53605, 100)
  (convs_list): ModuleList(
    (0): Conv2d (1, 128, kernel_size=(3, 100), stride=(1, 1))
    (1): Conv2d (1, 128, kernel_size=(5, 100), stride=(1, 1))
    (2): Conv2d (1, 128, kernel_size=(7, 100), stride=(1, 1))
  )
  (dropout): Dropout(p=0.2)
  (fc): Linear(in_features=384, out_features=4)
)
'''
if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr, weight_decay=0.01)
# train
cnn.train()
for epoch in range(1, args.epochs + 1):
    corrects, avg_loss = 0, 0
    for batch in train_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        optimizer.zero_grad()
        logit = cnn(feature)

        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(train_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('epoch[{}] Traning - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,
                                                                         avg_loss,
                                                                         accuracy,
                                                                         corrects,
                                                                         size))
    # test
    cnn.eval()
    corrects, avg_loss = 0, 0
    for batch in dev_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = cnn(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(dev_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,
                                                                   accuracy,
                                                                   corrects,
                                                                   size))
