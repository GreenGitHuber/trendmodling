from pca import PCA 
import numpy as np
import matplotlib.pyplot as plt

# 美国：12月22日到1月5日
#表格的时间02/01/2018 00:00:00

def use_pca(data):
    pca_obj = PCA(data,3)
    return pca_obj.main_x,pca_obj.rest_x

r=np.load("../data/17_12_pems_speed_occupancy_5min.npz")
speed_data=r["flow"]#speed_data.shape=(8928, 243),第二维243表示有243个sensor。
singel_sensor = speed_data[:,2]#这里因为index=2的sensor的shape符合M形状，所以取了这个sensor
m = singel_sensor.reshape(-1,288)  # 天数*288
data = m
print(data.shape)

data_main,data_rest=use_pca(data)  #shape 天数*288

for i in range(11,18):
    label = 'line+%d'%(i)
    plt.plot(data[i],label=label)
plt.legend(loc='best')
plt.show()
plt.close()

for i in range(11,18):
    label = 'line+%d'%(i)
    plt.plot(data_main[i],label=label)
plt.legend(loc='best')
plt.show()
plt.close()



plt.plot(data_main[0],label="data_main line")
plt.plot(data[1])
plt.legend(loc='upper right')
plt.show()
plt.close()
#the mean of reconstruct_main_x
mean_dmain=np.mean(data_main,axis=0)
plt.plot(mean_dmain)
plt.show()
plt.close()
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.set_ylim(-100,100)
for i in range(0,31):
    plt.plot(data_rest[i])
print(data_rest.shape)
plt.show()
plt.close()