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




# import matplotlib.pyplot as plt 
# from numpy.random import randn
# import numpy as np

# fig = plt.figure()#创建对象，因为matplotlib的图像都是位于figure对象中
# #不能通过空figure绘图，必须通过add_subplot创建一个或者是多个subplot

# ax1 = fig.add_subplot(2,2,1)#这条代码的意思是：图像应该是2*2的(即有2行2列，定义了图像的位置)，且当前选中的是第一个，(编号从一开始)
# ax2 = fig.add_subplot(2,2,2)#这条代码的意思是：图像应该是2*2的(即有2行2列，定义了图像的位置)，且当前选中的是第一个，(编号从一开始)
# ax3 = fig.add_subplot(2,2,3)#这条代码的意思是：图像应该是2*2的(即有2行2列，定义了图像的位置)，且当前选中的是第一个，(编号从一开始)

# ax1.hist(randn(100),bins=20,color='k',alpha=0.3) # 20个直方图,alpha透明度
# ax2.scatter(np.arange(30),np.arange(30)+3*randn(30)) # 绘制散点图
# plt.plot(randn(50).cumsum(),'k--')

# plt.show()
# plt.close()

# fig2 = plt.figure()#创建对象，因为matplotlib的图像都是位于figure对象中

# ax = fig2.add_subplot(1,1,1)
# ax.plot(randn(1000).cumsum())
# ticks = ax.set_xticks([0,250,500,750,1000])     # X轴刻度
# ax.set_title('My first matplotlib plot')    # 设置标题
# ax.set_xlabel('Stages')                     # 设置X轴名

# plt.show()

##############没有用figure的用法

# import numpy as np
# import matplotlib.pyplot as plt
 
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
 
# # 数据的直方图
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
 
 
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# #添加标题
# plt.title('Histogram of IQ')
# #添加文字
# plt.text(60, .025, r'$mu=100, sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
# plt.show()