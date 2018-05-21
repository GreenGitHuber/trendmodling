import numpy as np
import csv
from datetime import datetime

# no = [500011092, 500011102, 500011112, 500011113, 500014111, 500014112]
one_day_occupancy = np.zeros((288, 243))
one_day_speed = np.zeros((288, 243))
one_day_flow = np.zeros((288, 243))

def read_same_day(yearstep,year,m_str, d_str,sensor_id):
    occupancy = np.random.randn(288,243)
    speed = np.random.randn(288,243)
    flow = np.random.randn(288,243)

    for step in range(0,yearstep):
        y_str = str(year - step -1)
        try:
            csv_reader = csv.reader(
                open('../pems/%s/d05_text_station_5min_%s_%s_%s.txt' % (y_str,y_str, m_str, d_str)))
        except Exception as e:
            print('year '+ y_str +' month ' + m_str + ' day ' + d_str + ' does not exist!')
            continue
        i = 0  # i是one_year_occupancy的行数
        j = 0
        for row in csv_reader:
            # if datetime.strptime(row[0].split()[0],"%m/%d/%Y").weekday()+1 in [6,7]:
            #     break
            try:
                one_day_occupancy[i][j] = float(row[10])  # 从0开始
            except Exception as e:
                one_day_occupancy[i][j] = -1.0
            try:
                one_day_speed[i][j] = float(row[11])
            except Exception as e:
                one_day_speed[i][j] = one_day_speed[i][j-1]
            try:
                one_day_flow[i][j] = float(row[9])
            except Exception as e:
                    # one_day_speed[i][j] = -1.0
                one_day_flow[i][j] = one_day_flow[i][j-1]
            j += 1
            if j == 243:
                i += 1
                j = 0
        occupancy = np.vstack((occupancy, one_day_occupancy))
        speed = np.vstack((speed, one_day_speed))
        flow = np.vstack((flow, one_day_flow))
    return occupancy[288:,sensor_id].reshape(-1,288),speed[288:,sensor_id].reshape(-1,288),flow[288:,sensor_id].reshape(-1,288)

# year_occupancy,year_speed,year_flow= read_same_day(3,2017,'12','25')

# print(year_speed.shape)#(864, 243),一天是288，三天就是864

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