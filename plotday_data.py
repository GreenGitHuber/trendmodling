import matplotlib.pyplot as plt
import numpy as np
r=np.load("../data/pems_speed_occupancy_5min.npz")
speed_data=r["flow"]
singel_sensor = speed_data[:,2]
m = singel_sensor.reshape(53,-1)#这是一个sensor采集的53天数据
# for i in range(0,53):
#     plt.plot(m[i])
plt.plot(m[1])
plt.show()
