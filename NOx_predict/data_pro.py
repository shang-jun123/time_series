import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_excel(r'D:\git\time_series-master\NOx_predict\外三8预测数据.xlsx') #读取数据

df1=df1.iloc[0:30000,3:]#删除前两列没用的

df1[['入口Nox', '二次风和']] = df1[['二次风和', '入口Nox']]
df1.rename(index=str,columns={"入口Nox":"二次风和","二次风和":"入口Nox"},inplace=True)
print(df1.head())

import matplotlib.pyplot  as plt

lab=df1.columns[0:29]
for i in range(29):
    ybox=lab[i]
    plt.boxplot(df1[ybox],labels="s")
    plt.show()

