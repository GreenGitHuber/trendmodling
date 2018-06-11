import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split 
from pca import PCA

#预测两部分的数据
#主成分也进行预测，偏差数据也进行预测

def split_dataset(dataset,time_step):
    days,ndim = dataset.shape
    dataX=[]
    dataY=[]
    for i in range(0,days-time_step):
        dataX.append(dataset[i:i+time_step])
        dataY.append(dataset[i+time_step:i+time_step+1])
    return np.array(dataX),np.array(dataY)


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

r=np.load("../data/pems_speed_occupancy_5min.npz")
speed_data=r["flow"]
singel_sensor = speed_data[:,2]
m = singel_sensor.reshape(53,-1)  # 53*288
data = m

pca_obj = PCA(data,3)
data_main,data_rest=pca_obj.main_x,pca_obj.rest_x

dataset_rest = data_rest
dataset_main = data_main


# hyperparameters
batch = 50
test_size = 0.3
train_batch=int(batch * (1-test_size))
test_batch = int(batch*test_size)
n_inputs = 288
n_steps = 3             # time steps
n_hidden_units = 512    # neurons in hidden layer
n_output = 288
layer_num = 1
dropout_keep_rate = 0.9
max_epoch = int(2000 * 6)  # 6

# 归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaler_main = MinMaxScaler(feature_range=(0,1))

dataset_rest_scaler = scaler.fit_transform(dataset_rest)  # 53 288
dataset_main_scaler = scaler_main.fit_transform(dataset_main)  # 53 288


#
rest_dataX,rest_dataY = split_dataset(dataset_rest_scaler,time_step=n_steps)  #dataX shape (50,3,288) ,dataY shape (50,1,288)
main_dataX,main_dataY = split_dataset(dataset_main_scaler,time_step=n_steps)
rest_dataY=np.reshape(rest_dataY,(batch,n_output))                       #dataY shape (50,1,288)
main_dataY=np.reshape(main_dataY,(batch,n_output))                       #dataY shape (50,1,288)

#将daily data 和deviation data划分训练集和测试集
train_rest_X,test_rest_X,train_rest_y,test_rest_y =train_test_split(rest_dataX, rest_dataY, test_size=test_size, random_state=42)
train_main_X,test_main_X,train_main_y,test_main_y =train_test_split(main_dataX, main_dataY, test_size=test_size, random_state=42)



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


def main_lstm(layer_num, hidden_size, batch_size, output_size, lstm_x, keep_prob):
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        print("this")
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        # 我的版本是1.4 执行这一句，如果是版本小于0.12的话，那么执行上面的那一句
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)  # n_hidden_units = 128  neurons in hidden layer
    # lstm cell is divided into two parts (c_state, h_state) #batch_size =128
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(cell, lstm_x, initial_state=init_state, time_major=False)

    # outputs, state = tf.nn.dynamic_rnn(main_mlstm_cell, inputs=lstm_x, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
    lstm_y_pres = tf.layers.dense(h_state, output_size)
    return lstm_y_pres



# 根据输入数据来决定，train_num训练集大小,input_size输入维度
#train_num, time_step_size, input_size = dataX.shape  # sahpe ：12 * 2 *480
#_, output_size = dataY.shape

# **步骤1：LSTM 的输入shape = (batch_size, time_step_size, input_size)，输出shape=(batch_size, output_size)
rest_x_input = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
rest_y_real = tf.placeholder(tf.float32, [None, n_output])

main_x_input = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
main_y_real = tf.placeholder(tf.float32, [None, n_output])

# dropout的留下的神经元的比例
keep_prob = tf.placeholder(tf.float32, [])

# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

pre_layer_hidden_num = 0
pre_layer_hidden_size = 0
rest_hide_output = rest_x_input
main_hide_output = main_x_input

y_rest_pred = lstm(layer_num, n_inputs, batch_size, n_output, rest_hide_output, keep_prob)
y_main_pred = main_lstm(layer_num, n_inputs, batch_size, n_output, main_hide_output, keep_prob)


rest_loss=tf.reduce_mean(tf.square(tf.reshape(y_rest_pred,[-1])-tf.reshape(rest_y_real, [-1])))
rest_train_op = tf.train.AdamOptimizer(1e-3).minimize(rest_loss)

main_loss=tf.reduce_mean(tf.square(tf.reshape(y_main_pred,[-1])-tf.reshape(main_y_real, [-1])))
main_train_op = tf.train.AdamOptimizer(1e-3).minimize(main_loss)



sess1 = tf.Session()
sess2 = tf.Session()

# 初始化变量
sess1.run(tf.global_variables_initializer())
# sess2.run(tf.global_variables_initializer())
rest_cost_list=[]
iter_time = 100
for i in range(iter_time):
    feed_dict = {rest_x_input: np.array(train_rest_X), rest_y_real: np.array(train_rest_y), keep_prob: dropout_keep_rate, batch_size: train_batch}
    _,loss_rest=sess1.run([rest_train_op,rest_loss], feed_dict=feed_dict)
    print('iter:',i,'loss:',loss_rest)
    rest_cost_list.append(loss_rest)
rest_prob=sess1.run(y_rest_pred,feed_dict={rest_x_input:np.array(test_rest_X),rest_y_real: np.array(test_rest_y), keep_prob:1,batch_size:test_batch})


sess2.run(tf.global_variables_initializer())
main_cost_list=[]
iter_time = 100
for i in range(iter_time):
    feed_dict = {main_x_input: np.array(train_main_X), main_y_real: np.array(train_main_y), keep_prob: dropout_keep_rate, batch_size: train_batch}
    _,loss_main=sess2.run([main_train_op,main_loss], feed_dict=feed_dict)
    print('iter:',i,'loss:',loss_main)
    main_cost_list.append(loss_main)

main_prob=sess2.run(y_main_pred,feed_dict={main_x_input:np.array(test_main_X),main_y_real: np.array(test_main_y), keep_prob:1,batch_size:test_batch})

#反归一化
realpredict_rest_y, real_rest_y= scaler.inverse_transform(rest_prob), scaler.inverse_transform(test_rest_y)
realpredict_main_y, real_main_y= scaler_main.inverse_transform(main_prob), scaler_main.inverse_transform(test_main_y)

pre_reconstruct=pca_obj.reconstruct(realpredict_main_y,realpredict_rest_y)
real_reconstruct=pca_obj.reconstruct(real_main_y,real_rest_y)
print_res_index(pre_reconstruct, real_reconstruct,get_metrics)
       
        
plt.plot(pre_reconstruct[0],label='Predicted reconstruct line')
plt.plot(real_reconstruct[0],label='Real_flow line')
plt.plot(realpredict_rest_y[0],label='Predicted line')
plt.plot(real_rest_y[0],label='Excepted line')
plt.legend(loc='best')
plt.show()
plt.close()
plt.plot(rest_cost_list)
plt.title('Deviation Cost Curse')
plt.xlabel('Iters') 
plt.ylabel('Cost')
plt.show()
plt.close()
sess1.close()
sess2.close()


#
# mre: 0.00337021775097
# mae: 0.106004248284
# rmse: 0.157376649752
