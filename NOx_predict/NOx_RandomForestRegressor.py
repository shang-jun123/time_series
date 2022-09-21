import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_excel(r'D:\git\time_series\NOx_predict\NOxdata\NOxdata3w_pre.xlsx') #读取数据


df1=df1.iloc[0:30000,3:]#删除前两列没用的

df1[['入口Nox', '二次风和']] = df1[['二次风和', '入口Nox']]
print(df1.head())


from sklearn import preprocessing#进行归一化操作
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1.iloc[:,0:28])

df = pd.DataFrame(df0, columns=df1.columns[0:28])
df.loc[:, "二次风和"] = df1["二次风和"]
print(df.head())

cut=6000

train=df.iloc[:-cut,:]
test=df.iloc[-cut:,:]
test =test.reset_index(drop=True)

x_train=train.iloc[:,0:28]
y_train=train.iloc[:,28:29]
x_test=test.iloc[:,0:28]
y_test=test.iloc[:,28:29]

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



