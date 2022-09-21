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

from sklearn.neural_network import rbm

# Loading dataset
df=pd.read_csv(r"D:\git\time_series-master\LSTM_SVM_RF_time_series\NOx_predict\DBN\data.csv",encoding='gbk')
#X, Y = boston.data, boston.target
X=df.iloc[:,0:6]
Y=df.iloc[:,6:7]
Y=np.array(Y)
Y=Y.reshape(-1)
# Splitting data
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=20,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt
x=np.linspace(1,74,74)
# plt.scatter(x,Y_test)
# plt.scatter(x,Y_pred)

plt.plot(x,Y_test,label='Y_test')
plt.plot(x,Y_pred,label='Y_pred')
plt.legend(loc='upper right')
plt.show()


print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
