import torch.utils.data as data

import os
import os.path
import torch

import librosa
import numpy as np

# 音频数据格式，只允许wav和WAV
AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

# 判断是否是音频文件
def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

# 找到类名并索引
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# 构造数据集
def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects

# 频谱加载器， 处理音频，生成频谱
def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # 短时傅立叶变换
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)  # 计算幅度谱和相位

    # S = log(S+1)
    spect = np.log1p(spect) # 计算log域幅度谱

    # 处理所有的频谱，使得长度一致，少于规定长度，补0到规定长度; 多于规定长度的， 截短到规定长度;
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:max_len, ]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score 归一化
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect

# 音频加载器， 类似PyTorch的加载器，实现对数据的加载
class SpeechLoader(data.Dataset):
    """ Google 音频命令数据集的数据形式如下:
        root/one/xxx.wav
        root/head/123.wav
    参数:
        root (string): 原始数据集路径
        window_size:　STFT的窗长大小，默认参数是 .02
        window_stride: 用于STFT窗的帧移是 .01
        window_type: , 窗的类型，　默认是　hamming窗
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: 帧的最大长度
     属性:
        classes (list): 类别名的列表
        class_to_idx (dict): 　目标参数(class_name, class_index)(字典类型).
        spects (list):  频谱参数(spects path, class_index) 的列表
        STFT parameter: 窗长, 帧移, 窗的类型, 归一化
    """

    def __init__(self, root,  window_size=.02, window_stride=.01, window_type='hamming',
                 normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:  # 错误处理
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int):  序列
        Returns:
            tuple (spect, target):返回（spec,target）, 其中target 是类别的索引..
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)

        return spect, target

    def __len__(self):
        return len(self.spects)
