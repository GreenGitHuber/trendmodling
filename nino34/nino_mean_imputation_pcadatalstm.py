import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from pca import PCA
import json
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
    mre = np.mean(np.abs(y - pred_y) /np.abs(y))
    mae = np.mean(np.abs(y - pred_y))
    rmse = np.sqrt(np.mean(np.square(y-pred_y)))
    return mre,mae,rmse
def print_res_index(realY,predY,func):
    mre,mae,rmse = func(np.array(realY),np.array(predY))
    print('mre:',mre)
    print('mae:',mae)
    print('rmse:',rmse)
def flatten(x):
    result = []
    for el in x:
        result.extend(el)
    return result
def create_dataset(dataset,look_back=1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back):
        a = dataset[i :(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX), np.array(dataY).reshape(-1, 1)

np.random.seed(7)

with open(r"../../data/nino34imputation/means_imputation005.txt", encoding="utf-8") as f:
    d=json.load(f)

m = flatten(d)
data= np.array(d)
pca_obj = PCA(data,3)
data_main,data_rest=pca_obj.main_x,pca_obj.rest_x
data_rest = flatten(data_rest)
data_main = flatten(data_main)

data_rest = np.array(data_rest)
data_main = np.array(data_main)
data_main=data_main.reshape(-1,1)

dataset = data_rest

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.reshape(-1, 1))
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train_main, test_main = data_main[0:train_size,:], data_main[train_size:len(data_main),:]

# use this function to prepare the train and test datasets for modeling
look_back = 2
train_X, train_y = create_dataset(train, look_back)
test_X, test_y = create_dataset(test, look_back)
train_main_X, train_main_y = create_dataset(train_main, look_back)
test_main_X, test_main_y = create_dataset(test_main, look_back)

# reshape input to be [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], look_back,-1))
test_X = np.reshape(test_X, (test_X.shape[0], look_back, -1))
# # hyperparameters
batch = 1
n_inputs = 1
n_hidden_units = 256    # neurons in hidden layer
n_output = 1
layer_num = 1
dropout_keep_rate = 0.9
max_epoch = int(2000 * 6)  # 6


def lstm(layer_num, hidden_size, batch_size, output_size, lstm_x, keep_prob):
    def multi_cells(cell_num):
        # 多cell的lstm必须多次建立cell保存在一个list当中
        multi_cell = []
        for _ in range(cell_num):
            # **步骤2：定义LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

            # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            multi_cell.append(lstm_cell)
        return multi_cell

    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = rnn.MultiRNNCell(multi_cells(layer_num), state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # **步骤6：调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, time_step_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size]（中间的‘2’指的是每个cell中有两层分别是c和h）,
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=lstm_x, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # 输出层
    # W_o = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=0.1), dtype=tf.float32)
    # b_o = tf.Variable(tf.constant(0.1, shape=[output_size]), dtype=tf.float32)
    # y_pre = tf.add(tf.matmul(h_state, W_o), b_o)
    # tf.layers.dense是全连接层，不给激活函数，默认是linear function
    lstm_y_pres = tf.layers.dense(h_state, output_size)
    return lstm_y_pres
#
#
# 根据输入数据来决定，train_num训练集大小,input_size输入维度
#train_num, time_step_size, input_size = dataX.shape  # sahpe ：12 * 2 *480
#_, output_size = dataY.shape

# **步骤1：LSTM 的输入shape = (batch_size, time_step_size, input_size)，输出shape=(batch_size, output_size)
x_input = tf.placeholder(tf.float32, [None, look_back, n_inputs])
y_real = tf.placeholder(tf.float32, [None, n_output])

# dropout的留下的神经元的比例
keep_prob = tf.placeholder(tf.float32, [])

# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

pre_layer_hidden_num = 0
pre_layer_hidden_size = 0
hide_output = x_input
#
y_pred = lstm(layer_num, n_inputs, batch_size, n_output, hide_output, keep_prob)
# mse = tf.losses.mean_squared_error(y_real, y_pred)
loss=tf.reduce_mean(tf.square(tf.reshape(y_pred,[-1])-tf.reshape(y_real, [-1])))
# train_op = tf.train.AdamOptimizer(1e-3).minimize(mse)
train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)


sess = tf.Session()

# 初始化变量
sess.run(tf.global_variables_initializer())


cost_list=[]
iter_time = 100
for i in range(iter_time):
    feed_dict = {x_input: np.array(train_X), y_real: np.array(train_y), keep_prob: dropout_keep_rate, batch_size: train_size-look_back}
    # sess.run(train_op, feed_dict=feed_dict)
    _,loss_=sess.run([train_op,loss], feed_dict=feed_dict)
    # cost_=sess.run(mse, feed_dict=feed_dict)
    # print('iter:',i,'cost:',sess.run(mse, feed_dict=feed_dict))
    print('iter:',i,'loss:',loss_)
    cost_list.append(loss_)


prob=sess.run(y_pred,feed_dict={x_input:np.array(test_X),y_real: np.array(test_y), keep_prob:1,batch_size:test_size-look_back})
#反归一化
realpredict_y, real_y= scaler.inverse_transform(prob), scaler.inverse_transform(test_y)

pre_reconstruct=pca_obj.reconstruct(test_main_y,realpredict_y)
real_reconstruct=pca_obj.reconstruct(test_main_y,real_y)


print_res_index(pre_reconstruct, real_reconstruct,get_metrics)

plt.plot(pre_reconstruct,label='Predicted line')
plt.plot(real_reconstruct,label='Excepted line')
plt.show()
plt.close()


plt.plot(m,label='Excepted line')
plt.plot(([None for _ in range(train_size)] + [x for x in real_reconstruct]),label='Predicted line')
plt.title('RawData lstm predict')
plt.legend(loc='best')
plt.show()
plt.close()
plt.plot(cost_list)
plt.title('Train Cost Curse')
plt.xlabel('Iters')
plt.ylabel('Cost')
plt.show()
plt.close()

# mre: 0.005140510057390546
# mae: 0.1391943878887028
# rmse: 0.21463319958795363