"""
搭建一个简单的神经网络，借助MNIST数据集进行分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
import torchvision
import numpy as np
import warnings
warnings.filterwarnings('ignore') # 忽略警告

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数设置
EPOCH = 20  # 训练次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False  # 是否下载数据集，如果数据集下载好了就写False

def get_data(train_ds, test_ds):
    """打包数据集"""
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False)
    )

def get_model():
    """获取模型和优化器"""
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=LR)

def fit(epochs, model, loss_func, opt, train_dl, test_dl):
    """训练模型"""
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl] 
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('epoch:'+str(epoch), 'loss:'+str(val_loss))

def loss_batch(model, loss_func, xb, yb, opt=None):
    """计算损失"""
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='Rookie\part_2_items\MachineLearning\PyTorch\data',  # 下载文件的保存路径或数据提取路径
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='Rookie\part_2_items\MachineLearning\PyTorch\data',
    train=False  # 表明是测试集
)

# 建立自己的网络
class Mnist_NN(nn.Module): # 我们建立的Mnist_NN继承nn.Module这个模块
    def __init__(self): 
        super().__init__() # 调用父类的构造函数
        self.hidden1 = nn.Linear(28 * 28, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x
    
net = Mnist_NN() # 实例化网络
print(net) # 打印网络结构
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

train_ds = TensorDataset(train_data.train_data.view(-1, 28 * 28).type(torch.float32) / 255, train_data.train_labels) 
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = TensorDataset(test_data.test_data.view(-1, 28 * 28).type(torch.float32) / 255, test_data.test_labels)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False)

# main
train_dl, test_dl = get_data(train_ds, test_ds)
model, opt = get_model()
loss_func = F.cross_entropy
fit(EPOCH, model, loss_func, opt, train_dl, test_dl)

print(train_dl.dataset.tensors[0].shape)
print(test_dl.dataset.tensors[0].shape)