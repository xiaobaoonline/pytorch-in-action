# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 参数设置
torch.manual_seed(1)
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
N_EPHCNS = 100

# 语料
test_sentence = """Word embeddings are dense vectors of real numbers, 
one per word in your vocabulary. In NLP, it is almost always the case 
that your features are words! But how should you represent a word in a
computer? You could store its ascii character representation, but that
only tells you what the word is, it doesn’t say much about what it means 
(you might be able to derive its part of speech from its affixes, or properties 
from its capitalization, but not much). Even more, in what sense could you combine 
these representations?""".split()

# 三元模型语料准备
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
idx_to_word = {word_to_ix[word]: word for word in word_to_ix}


# 语言模型
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


# loss 函数和优化器
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练语言模型
for epoch in range(N_EPHCNS):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        # Step 1. 准备数据
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2 梯度初始化
        model.zero_grad()

        # Step 3. 前向算法
        log_probs = model(context_var)

        # Step 4. 计算loss
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. 后向算法和更新梯度
        loss.backward()
        optimizer.step()

        # step 6. loss
        total_loss += loss.data
    # 打印 loss
    print('\r epoch[{}] - loss: {:.6f}'.format(epoch, total_loss[0]))

word, label = trigrams[3]
word = autograd.Variable(torch.LongTensor([word_to_ix[i] for i in word]))
out = model(word)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.data[0]]
print('real word is {}, predict word is {}'.format(label, predict_word))

#
# epoch[91] - loss: 243.199814
#  epoch[92] - loss: 241.579529
#  epoch[93] - loss: 239.956345
#  epoch[94] - loss: 238.329926
#  epoch[95] - loss: 236.701630
#  epoch[96] - loss: 235.069275
#  epoch[97] - loss: 233.434341
#  epoch[98] - loss: 231.797974
#  epoch[99] - loss: 230.158493
#
