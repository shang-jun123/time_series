import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator

myfont = font_manager.FontProperties(fname="C:\Windows\Fonts\simfang.ttf")
plt.rcParams["font.sans-serif"] = "SimHei"  # 修改字体的样式可以解决标题中文显示乱码的问题
plt.rcParams["axes.unicode_minus"] = False  # 该项可以解决绘图中的坐标轴负数无法显示的问题

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn.tensorflow import SupervisedDBNRegression
import pandas as pd

import xlrd

from sklearn.neural_network import rbm


df1=pd.read_excel(r'D:\git\time_series-master\LSTM_SVM_RF_time_series\NOx_predict\NOxdata3w_pre.xlsx') #读取数据
#df1=pd.read_excel(r'D:\git\time_series-master\LSTM_SVM_RF_time_series\NOx_predict\外三8预测数据.xlsx') #读取数据

df1=df1.iloc[0:30000,3:]#删除前两列没用的

df1[['入口Nox', '二次风和']] = df1[['二次风和', '入口Nox']]
print(df1.head())


from sklearn import preprocessing#进行归一化操作
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1.iloc[:,0:29])

df = pd.DataFrame(df0, columns=df1.columns[0:29])
#df.loc[:, "二次风和"] = df1["二次风和"]
print(df.head())

cut=6000

train=df.iloc[:-cut,:]
test=df.iloc[-cut:,:]
test =test.reset_index(drop=True)

X_train=train.iloc[:,0:28]
Y_train=train.iloc[:,28:29]
Y_train=np.array(Y_train)
Y_train=Y_train.reshape(-1)

X_test=test.iloc[:,0:28]
Y_test=test.iloc[:,28:29]
Y_test=np.array(Y_test)
Y_test=Y_test.reshape(-1)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=2,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)

#在训练集上的拟合结果
X_train= min_max_scaler.transform(X_train)
Y_pred = regressor.predict(X_train)

y_train_predict=Y_pred
y_train=Y_train
draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[100:20000,0].plot(figsize=(12,6))
draw.iloc[100:20000,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题
plt.show()


# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)


y_test_predict=Y_pred
y_test=Y_test

draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题
plt.show()

# import matplotlib.pyplot as plt
#
# x = np.linspace(1, 74, 74)
# # plt.scatter(x,Y_test)
# # plt.scatter(x,Y_pred)
#
# plt.plot(x, Y_test, label='Y_test')
# plt.plot(x, Y_pred, label='Y_pred')
# plt.legend(loc='upper right')
# plt.show()

print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
