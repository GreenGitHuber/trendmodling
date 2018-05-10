from pca import PCA 
import numpy as np
import matplotlib.pyplot as plt


def use_pca(data):
    pca_obj = PCA(data,3)
    return pca_obj.main_x,pca_obj.rest_x

r=np.load("../data/pems_speed_occupancy_5min.npz")
speed_data=r["flow"]
singel_sensor = speed_data[:,2]
m = singel_sensor.reshape(53,-1)  # 53*288
data = m

data_main,data_rest=use_pca(data)  #shape 53*288

plt.plot(data_main[0])
#the mean of reconstruct_main_x
mean_dmain=np.mean(data_main,axis=0)
plt.plot(mean_dmain)
plt.show()
plt.close()
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.set_ylim(-100,100)
for i in range(0,53):
    plt.plot(data_rest[i])
plt.show()
plt.close()