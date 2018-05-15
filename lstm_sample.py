import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings 
from datahelp import split_train_test_dataset,generate_data
from pca import PCA 

def use_pca(data):
    pca_obj = PCA(data,3)
    return pca_obj.main_x,pca_obj.rest_x

r=np.load("../data/pems_speed_occupancy_5min.npz")
speed_data=r["flow"]
singel_sensor = speed_data[:,2]
m = singel_sensor.reshape(53,-1)  # 53*288
data = m

data_main,data_rest=use_pca(data)

data = data_rest

#定义常量
rnn_unit =10 #hidden layer units
input_size =288
output_size = 288
lr = 0.0006
tf.reset_default_graph()
#输入层，输出层权重、偏置
weights={
    'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
    'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
}
biases = {
    'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
    'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
}



#分割数据集，将数据分为训练集和验证集（最后90天做验证，其他做训练）：
def get_data(batch_size=80,time_step=7,ration=0.6):
    batch_index=[]
    # scaler_for_data=MinMaxScaler(feature_range=(0,1))  #按列做minmax缩放
    # scaler_for_y=MinMaxScaler(feature_range=(0,1))  
    # scaled_data=scaler_for_data.fit_transform(data)
    # scaled_x_data=scaler_for_x.fit_transform(data[:,:-1])
    # scaled_y_data=scaler_for_y.fit_transform(data[:,-1][:,np.newaxis]) # data[:,-1]是行向量，加上[:,np.newaxis]就变成了列向量
    
    train_data_set,test_data_set=split_train_test_dataset(data,ration)#划分数据为训练集和测试集
    train_x_data,train_y_data = generate_data(train_data_set,time_step)
    test_x_data,test_y_data = generate_data(test_data_set,time_step)#(50,7,288) ,dataY shape (50,1,288)

    label_train = train_y_data 
    label_test = test_y_data
    normalized_train_data = train_x_data
    normalized_test_data = test_x_data 
    
    train_x,train_y=[],[]   #训练集x和y初定义
    for i in range(len(normalized_train_data)): 
        if i % batch_size==0:  
            batch_index.append(i)

        # x=normalized_train_data[i:i+1]
        # y=label_train[i:i+1]
        # train_x.append(x.tolist())
        # train_y.append(y.tolist())
    # batch_index.append((len(normalized_train_data)-time_step))
    num = (len(normalized_test_data)+time_step-1)//time_step
    test_x,test_y=[],[] 
    # for i in range(len(normalized_test_data)):
    #     x=normalized_test_data[i:i+1]
    #     y=label_test[i:i+1]
    #     test_x.append(x.tolist())
    #     test_y.extend(y.tolist())
    return train_x_data,train_y_data,test_x_data,test_y_data

#——————————————————定义神经网络变量—————————————————— 
def lstm(X):#X.shape=batch_size*time_step*input_size
    batch_size=tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in=weights['in']
    b_in = biases['in']
    input = tf.reshape(X,[-1,input_size])#需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn = tf.reshape(input_rnn,[-1,time_step,rnn_unit])#将tensor转成3维，作为lstm cell的输入  ；rnn_unit=10
    #TensorFlow的输入形式是（Batch_size，num_step，embeding_size）
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)#rnn_unit =10 #hidden layer units
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell,input_rnn,initial_state=init_state, dtype=tf.float32)
    #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out#这里是输出层，这里可以改变，比如用sigmod函数
    return pred,final_states

#——————————————————训练模型——————————————————  
def train_lstm(batch_size=80,time_step=7,ration=0.6):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,1,output_size])
    train_x,train_y,test_x,test_y = get_data(batch_size,time_step,ration)
    pred,_=lstm(X)
    
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())  
        #重复训练5000次  
        iter_time = 5000
        for i in range(iter_time):  
            for step in range(len(train_x)-batch_size+1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[step*batch_size:(step+1)*batch_size],Y:train_y[step*batch_size:(step+1)*batch_size]})

            if i % 100 == 0:
               print('iter:',i,'loss:',loss_)
        ####predict####  
        test_predict=[]  
        for step in range(len(test_x)):  
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})     
            predict=prob.reshape((-1,1))  
            test_predict.extend(predict)  
            
        # test_predict = scaler_for_y.inverse_transform(test_predict)  
        # test_y = scaler_for_y.inverse_transform(test_y)  
        rmse=np.sqrt(mean_squared_error(test_predict,test_y))  
        mae = mean_absolute_error(y_pred=test_predict,y_true=test_y)  
        print ('mae:',mae,'   rmse:',rmse)  
    return test_predict 

test_predict = train_lstm(batch_size=10,time_step=7,ration=0.6) 
