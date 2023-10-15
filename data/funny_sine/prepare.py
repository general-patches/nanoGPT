from math import sin
import matplotlib.pyplot as plt
import numpy as np
import os

data = [int(65*sin(i*0.1)/2) + 65/2 for i in range(10000)]

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_data = np.array(train_data, dtype=np.uint16)
val_data = np.array(val_data, dtype=np.uint16)
train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

plt.plot(data)
plt.show()