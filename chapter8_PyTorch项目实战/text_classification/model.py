import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        Ci = 1
        kernel_num = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(embed_num, embed_dim)

        self.convs_list = nn.ModuleList(
            [nn.Conv2d(Ci, kernel_num, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_list]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return logit
