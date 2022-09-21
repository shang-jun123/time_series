import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#df1=pd.read_excel(r'D:\git\time_series\LSTM_SVM_RF_time_series\NOx_predict\外三8预测数据_17.xlsx') #读取数据
df1=pd.read_excel(r'D:\git\time_series\LSTM_SVM_RF_time_series\NOx_predict\数据拼接（剔除空白值）.xlsx') #读取数据

df1=df1.iloc[:,1:]#删除前两列没用的

df1[['脱硝A侧NOX', 'B空预器入口烟气含氧量']] = df1[['B空预器入口烟气含氧量', '脱硝A侧NOX']]
print(df1.head())

print(np.isnan(df1).any())
df1.dropna(inplace=True)

from sklearn import preprocessing#进行归一化操作
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1.iloc[:,0:57])

df = pd.DataFrame(df0, columns=df1.columns[0:57])
df.loc[:, "B空预器入口烟气含氧量"] = df1["B空预器入口烟气含氧量"]
print(df.head())

print(np.isnan(df).any())
df.dropna(inplace=True)

cut=10000
train=df.iloc[:-cut,:]
test=df.iloc[-cut:,:]
test =test.reset_index(drop=True)

x_train=train.iloc[:,0:57]
y_train=train.iloc[:,57:58]
x_test=test.iloc[:,0:57]
y_test=test.iloc[:,57:58]

# from sklearn.svm import SVR
# svr = SVR(kernel='rbf', gamma=0.1)
# #model = svr.fit(X_train, y_train)
# model = svr.fit(x_train, y_train)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model = rf.fit(x_train, y_train)

#在训练集上的拟合结果
y_train_predict=model.predict(x_train)


draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[100:20000,0].plot(figsize=(12,6))
draw.iloc[100:20000,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题
plt.show()

y_test_predict=model.predict(x_test)

draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题
plt.show()



