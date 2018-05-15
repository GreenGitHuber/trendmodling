"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

His code is a very good one for RNN beginners. Feel free to check it out.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pylab as plt
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

dataset = data

def split_dataset(dataset,time_step):
    days,ndim = dataset.shape
    dataX=[]
    dataY=[]
    for i in range(0,days-time_step):
        dataX.append(dataset[i:i+time_step])
        dataY.append(dataset[i+time_step:i+time_step+1])
        #print ("x ",i," ",i+time_step)
        #print ("y ",i+time_step," ",i+time_step+1)
    return np.array(dataX),np.array(dataY)





# hyperparameters
lr = 0.001
training_iters = 1000
batch_size = 50

n_inputs = 288   # MNIST data input (img shape: 28*28)
n_steps = 3    # time steps
n_hidden_units = 512   # neurons in hidden layer
#n_classes = 10      # MNIST classes (0-9 digits)
n_output = 288


# dataset = np.random.rand(53,288)  # shape 53 * 288

#归一化
scaler = MinMaxScaler(feature_range=(0,1))
dataset_scaler = scaler.fit_transform(dataset)

#
dataX,dataY = split_dataset(dataset_scaler,time_step=n_steps)  #dataX shape (50,3,288) ,dataY shape (50,1,288)
dataY=np.reshape(dataY,(batch_size,n_output))                  #dataY shape (50,1,288)



# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_output])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_output]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_output, ]))
}


def lstm(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    X_in = tf.matmul(X, weights['in']) + biases['in']  # (128*28, 28 ) * (28,128) = (128*28,128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        print("this")
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        # 我的版本是1.4 执行这一句，如果是版本小于0.12的话，那么执行上面的那一句
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)  # n_hidden_units = 128  neurons in hidden layer
    # lstm cell is divided into two parts (c_state, h_state) #batch_size =128
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results


pred = lstm(x, weights, biases)
cost = tf.losses.mean_squared_error(y, pred)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)


with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    iter_time = 5000
    for i in range(iter_time):
        sess.run([train_op], feed_dict={x: dataX,y: dataY})
        cost_=sess.run(cost,feed_dict={x: dataX,y: dataY})
        if i>700 and i % 100==0:
            print('iter:',i,'cost:',cost_)
            # y_pre = sess.run(pred,feed_dict={x: dataX,y: dataY})  #(50,288)
            # plt.plot(y_pre[0],label='Predicted line')
            # plt.plot(dataY[0],label='Expected line')
            # plt.legend(loc='upper right')
            # plt.show()
            plt.close()