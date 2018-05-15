import matplotlib.pyplot as plt
import numpy as np
# r=np.load("../data/pems_speed_occupancy_5min.npz")
# flow_data=r["flow"]
# singel_sensor = flow_data[:,2]
# m = singel_sensor.reshape(53,-1)


def split_train_test_dataset(dataset,ration):
    lenth = dataset.shape[0]
    train_len = int(lenth*ration)
    train_data_set = dataset[0:train_len]
    test_data_set = dataset[train_len:lenth]
    return train_data_set,test_data_set

def generate_data(dataset,time_step):
    days,ndim = dataset.shape
    dataX=[]
    dataY=[]
    for i in range(0,days-time_step):
        dataX.append(dataset[i:i+time_step])
        dataY.append(dataset[i+time_step:i+time_step+1])
        # print ("x ",i," ",i+time_step)
        # print ("y ",i+time_step," ",i+time_step+1)
    return np.array(dataX),np.array(dataY)


# dataset = np.random.rand(53,288)  # shape 53 * 288
#
# dataX,dataY = generate_data(dataset,3)  #dataX shape (50,3,288) ,dataY shape (50,1,288)
# test_X = dataX[0:3]
# print(test_X.shape)