from pca import PCA 
import numpy as np
import matplotlib.pyplot as plt


def use_pca(data):
    pca_obj = PCA(data,3)
    return pca_obj.main_x,pca_obj.rest_x

r=np.load("../data/pems_speed_occupancy_5min.npz")
speed_data=r["flow"]
singel_sensor = speed_data[:,2]
m = singel_sensor.reshape(-1,288)  # 361*288
data = m

data_main,data_rest=use_pca(data)  #shape 361*288

# for i in range(0,361):
#     plt.plot(data[i])
# plt.show()
# plt.close()

# for i in range(0,361):
#     plt.plot(data_main[i])
# plt.show()
# plt.close()

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
for i in range(0,53):
    plt.plot(data_rest[i])
print(data_rest.shape)
plt.show()
plt.close()