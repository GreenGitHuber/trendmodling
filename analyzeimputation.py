from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split 
from pca import PCA
import json
import matplotlib.mlab as mlab

#将用ppca补全的数据进行预测。
#偏差数据进行预测。
#加上实际的主成分。

def split_dataset(dataset,time_step):
    days,ndim = dataset.shape
    dataX=[]
    dataY=[]
    for i in range(0,days-time_step):
        dataX.append(dataset[i:i+time_step])
        dataY.append(dataset[i+time_step:i+time_step+1])
    return np.array(dataX),np.array(dataY)


def use_pca(data):
    pca_obj = PCA(data,3)
    return pca_obj.main_x,pca_obj.rest_x

def get_metrics(y,pred_y):
    y_mean=np.mean(y)
    y[y==0.00]=y_mean
    mre = np.mean(np.abs(y - pred_y) / np.abs(y))
    mae = np.mean(np.abs(y - pred_y))
    rmse = np.sqrt(np.mean(np.square(y-pred_y)))
    return mre,mae,rmse

def print_res_index(realY,predY,func):
    mre,mae,rmse = func(np.array(realY),np.array(predY))
    print('mre:',mre)
    print('mae:',mae)
    print('rmse:',rmse)
# f= open("../data/inputationdata/ppca_imputation0050000.txt",'rb')
with open(r"../data/imputationdata/ppca_my_imputation005.txt", encoding="utf-8") as f:
    d=json.load(f)
speed_data=np.array(d)
m = speed_data.reshape(53,-1)  # 53*288
data = m

pca_obj = PCA(data,3)
data_main,data_rest=pca_obj.main_x,pca_obj.rest_x

def drew_hist(lengths):
    data = lengths
    mu = np.mean(data)
    sigma = np.std(data)
    n,bins,patches = plt.hist(data,60,normed=1,histtype='bar',facecolor='darkblue',alpha=0.75)
    plt.title('The distribution of the residuals')
    plt.xlabel('Probability density')
    y = mlab.normpdf(bins,mu,sigma)
    plt.plot(bins,y,'r')
    plt.show()
drew_hist(data_rest[8])




#imputation
# mre: 0.016490104409841277
# mae: 0.4975384880057572
# rmse: 1.1250220556432569

#
# mre: 0.010025252878355624
# mae: 0.30091390294632925
# rmse: 0.5876020455325003