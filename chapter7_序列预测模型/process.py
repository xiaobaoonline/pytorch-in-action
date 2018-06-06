#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : process.py
# @Time    : 18-3-14
# @Author  : J.W.

from __future__ import unicode_literals, print_function, division

import math
import re
import time
import unicodedata

import jieba
import torch
from logger import logger
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
# 中文的时候要设置大一些
MAX_LENGTH = 25


def unicodeToAscii(s):
    '''
    Unicode转换成ASCII，http://stackoverflow.com/a/518232/2809427
    :param s:
    :return:
        '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    '''
    转小写，去除非法字符
    :param s:
    :return:
    '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # 中文不能进行下面的处理
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Lang:
    def __init__(self, name):
        '''
        添加 need_cut 可根据语种进行不同的分词逻辑处理
        :param name: 语种名称
        '''
        self.name = name
        self.need_cut = self.name == 'cmn'
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # 初始化词数为2：SOS & EOS

    def addSentence(self, sentence):
        '''
        从语料中添加句子到 Lang
        :param sentence: 语料中的每个句子
        '''
        if self.need_cut:
            sentence = cut(sentence)
        for word in sentence.split(' '):
            if len(word) > 0:
                self.addWord(word)

    def addWord(self, word):
        '''
        向 Lang 中添加每个词，并统计词频，如果是新词修改词表大小
        :param word:
        '''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def cut(sentence, use_jieba=False):
    '''
    对句子分词。
    :param sentence: 要分词的句子
    :param use_jieba: 是否使用 jieba 进行智能分词，默认按单字切分
    :return: 分词结果，空格区分
    '''
    if use_jieba:
        return ' '.join(jieba.cut(sentence))
    else:
        words = [word for word in sentence]
        return ' '.join(words)


def readLangs(lang1, lang2, reverse=False):
    '''

    :param lang1: 源语言
    :param lang2: 目标语言
    :param reverse: 是否逆向翻译
    :return: 源语言实例，目标语言实例，词语对
    '''
    logger.info("Reading lines...")

    # 读取txt文件并分割成行
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # 按行处理成 源语言-目标语言对，并做预处理
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    '''
    按自定义最大长度过滤
    '''
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    logger.info("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    logger.info("Trimmed to %s sentence pairs" % len(pairs))
    logger.info("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    logger.info("Counted words:")
    logger.info('%s, %d' % (input_lang.name, input_lang.n_words))
    logger.info('%s, %d' % (output_lang.name, output_lang.n_words))
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    '''
    :param lang:
    :param sentence:
    :return:
    '''
    return [lang.word2index[word] for word in sentence.split(' ') if len(word) > 0]


def variableFromSentence(lang, sentence):
    if lang.need_cut:
        sentence = cut(sentence)
    # logger.info("cuted sentence: %s" % sentence)
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(input_lang, output_lang, pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
