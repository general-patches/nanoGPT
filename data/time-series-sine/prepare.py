from math import sin
import matplotlib.pyplot as plt
from statistics import fmean, stdev
import numpy as np
import os

data = [sin(i*0.1) for i in range(10000)]

n = len(data)
train_data = data[:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.85)]
test_data = data[int(n*0.85):]

mean = fmean(train_data)
std = stdev(train_data)
for index in range(len(train_data)):
    train_data[index] = (train_data[index] - mean) / std

for index in range(len(val_data)):
    val_data[index] = (val_data[index] - mean) / std

for index in range(len(test_data)):
    test_data[index] = (test_data[index] - mean) / std

with open(os.path.join(os.path.dirname(__file__), 'meta.txt'), 'w') as f:
    f.write(f"mean={mean}\nstd={std}")

train_data = np.array(train_data, dtype=np.float32)
val_data = np.array(val_data, dtype=np.float32)
test_data = np.array(test_data, dtype=np.float32)
train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_data.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

plt.plot(test_data)
plt.show()
