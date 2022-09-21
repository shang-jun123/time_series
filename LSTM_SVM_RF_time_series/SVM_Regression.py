import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
#倒入一些必要的库

feanum=6#一共有多少特征
window=5#时间窗设置
#df1=pd.read_csv('trend.csv') #读取数据
df1=pd.read_excel(r'天气数据.xlsx') #读取数据
#print(df1.head())
df1=df1.iloc[:,2:]#删除前两列没用的
df1.tail()
from sklearn import preprocessing#进行归一化操作
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
df.tail()
print(df.head())

#这一部分在处理数据 将原始数据改造为LSTM网络的输入
stock=df
seq_len=window
amount_of_features = len(stock.columns)#有几列

#data = stock.as_matrix() #pd.DataFrame(stock) 表格转化为矩阵
data = stock.iloc[:,:].values
sequence_length = seq_len + 1#序列长度+1
result = []
for index in range(len(data) - sequence_length):#循环 数据长度-时间窗长度 次
    result.append(data[index: index + sequence_length])#第i行到i+5
result = np.array(result)#得到样本，样本形式为 window*feanum
cut=150#分训练集测试集 最后cut个样本为测试集
train = result[:-cut, :]
x_train = train[:, :-1]
y_train = train[:, -1][:,-1]
x_test = result[-cut:, :-1]
y_test = result[-cut:, -1][:,-1]
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

print(x_train)
#展示下训练集测试集的形状 看有没有问题
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

#建立、训练模型过程
# Fit regression model

nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))

from sklearn.svm import SVR
svr = SVR(kernel='rbf', gamma=0.1)
#model = svr.fit(X_train, y_train)
model = svr.fit(d2_train_dataset, y_train)

#在训练集上的拟合结果
y_train_predict=model.predict(d2_train_dataset)

draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[100:400,0].plot(figsize=(12,6))
draw.iloc[100:400,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题
plt.show()
#展示在训练集上的表现

#在测试集上的预测
nsamples, nx, ny = X_test.shape
d2_test_dataset =X_test.reshape((nsamples,nx*ny))


y_test_predict=model.predict(d2_test_dataset)

draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题
plt.show()
#展示在测试集上的表现