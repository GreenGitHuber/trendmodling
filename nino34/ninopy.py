import numpy as np
import matplotlib.pyplot as plt

def flatten(x):
    result = []
    for el in x:
        result.extend(el)
    return result

a = np.loadtxt('../nino34/nino34.long.anom.data.txt')

a = a[:-1,1:]

one_dim_data = flatten(a)
print(one_dim_data)
plt.plot(one_dim_data)
plt.show()
plt.close()