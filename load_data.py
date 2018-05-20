import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

data = []
label = []
path = "slices/"
for root, _, files in os.walk(path):
    for name in files:
        f = open(os.path.join(root, name), encoding="utf-8")
        tmp = f.read()
        data.append(tmp)
        label.append(int(name[-5]))


pos = []
neg = []
for i in range(len(data)):
    if label[i] == 0:
        pos.append(data[i])
    else:
        neg.append(data[i])

p_sample = random.sample(pos, 500)
n_sample = random.sample(neg, 500)

pairs = []
pair_labels = []
for i in range(100):
    x1 = random.choice(n_sample)
    y1 = random.choice(n_sample)
    pairs.append([x1, y1])
    pair_labels.append(1)

for i in range(100):
    x1 = random.choice(n_sample)
    y1 = random.choice(p_sample)
    pairs.append([x1, y1])
    pair_labels.append(0)

seed = 0
np.random.seed(seed)
np.random.shuffle(pairs)
np.random.seed(seed)
np.random.shuffle(pair_labels)

x_train, x_test, y_train, y_test = train_test_split(pairs, pair_labels, test_size=0.1, random_state=0)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(len(x_train[0]))
print(y_train[0])

save_path = "pairs.npz"
np.savez(save_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)