import matplotlib.pyplot as plt
import numpy as np
import os

data = []
with open(os.path.join(os.path.dirname(__file__), 'input.txt')) as f:
    for val in f.readlines()[0].split(','):
        data.append(int(val))

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_data = np.array(train_data, dtype=np.float32)
val_data = np.array(val_data, dtype=np.float32)
train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

plt.plot(val_data)
plt.show()
