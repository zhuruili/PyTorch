# env: Python 3.9.18
# 基于LSTM(长短期记忆网络)的飞机航班人数预测
# 源教程地址(英文)：https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# Tip：原教程为CPU版本，本程序在原教程基础上增加了GPU支持，并支持自动选择设备进行训练，并扩展了模型保存功能
"""
Warning:
1. 如果出现了绘图，需要关闭绘图窗口后程序才能继续执行
2. 如果发现程序每次运行结果不一致为正常现象，因为PyTorch的初始权重设置具有随机性，如果想限定结果可以尝试设置随机种子
3. 本程序使用的数据集为seaborn自带的‘flights’数据集，如果报错TimeoutError等，请检查网络环境，必要时可开启加速器
4. 运行程序之前请确保您已经配置好了运行程序所需的环境并安装了对应的包
"""
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

# 超参数设置
test_data_size = 12 # 测试集大小
train_window = 12 # 训练输入序列长度
input_size = 1 # 输入数据维度
hidden_layer_size = 100 # 隐藏层神经元数量
output_size = 1 # 输出数据维度
lr = 0.001 # 学习率
epochs = 125 # 训练轮数
fut_pred = 12 # 预测数据长度
save_model = False # 是否保存模型

# 功能函数
def create_inout_sequences(input_data, tw):
    """
    创建输入输出序列
    input_data: 输入数据
    tw: 训练输入序列长度
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# print(sns.get_dataset_names()) # 获取seaborn自带数据集，本程序使用其中的‘flights’数据集
flight_data = sns.load_dataset("flights") # 加载‘flights’数据集
# print(flight_data.head()) # 显示数据集的前5行，这里数据包含年份、月份和乘客数量
# print(flight_data.shape) # 显示数据集的行数和列数，数据规模为(144, 3)

"""
任务：根据已知的乘客数据预测未来乘客数量
数据集的使用：前132个月的数据用于训练，后12个月的数据用于测试
"""

# 绘制乘客数量随时间变化的折线图观察基本特征
# 设置图像尺寸
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
# 设置样式
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True) # 显示网格
plt.autoscale(axis='x', tight=True) # 自动调整x轴
# 绘制折线图
plt.plot(flight_data['passengers'])
plt.show() # 显示图形

# 数据处理
all_data = flight_data['passengers'].values.astype(float) # 将object类型转换为float类型
# print(all_data) # 打印数据
# 划分训练集与测试集
train_data = all_data[:-test_data_size] # 训练集
test_data = all_data[-test_data_size:] # 测试集
# print(len(train_data), len(test_data)) 
# 归一化处理
scaler = MinMaxScaler(feature_range=(-1, 1)) # 将数据缩放到[-1, 1]之间
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1)) # 训练集归一化
# data to tensor，将数据转换为张量，在机器学习领域中数据的表现形式通常并不是我们熟悉的int、float等，而是一种叫做张量的数据结构
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1) 

train_inout_seq = create_inout_sequences(train_data_normalized, train_window) # 创建输入输出序列

# 创建LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size) # 创建LSTM模型对象
loss_function = nn.MSELoss() # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 优化器
# print(model) # 打印模型结构

# 检测CUDA是否可用，并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')
model.to(device)

# 训练模型
print('------ Start training ------')
for i in range(epochs):
    for seq, labels in train_inout_seq:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad() # 梯度清零
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq) # 预测

        single_loss = loss_function(y_pred, labels) # 计算损失
        single_loss.backward() # 反向传播
        optimizer.step() # 优化

    if i%25 == 1: # 每25轮输出一次训练结果
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}') # 输出最终训练结果
print('------ End training ------')

# 保存模型
if save_model:
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved!')

# 使用模型进行预测
test_inputs = train_data_normalized[-train_window:].tolist() # 测试输入

model.eval() 

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_inputs.append(model(seq).item())

# 反归一化
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
# print(actual_predictions)

# 绘制预测结果并与实际结果对比
x = np.arange(132, 144, 1) # x轴数据
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x, actual_predictions)
plt.show()
