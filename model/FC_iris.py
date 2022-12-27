import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import metatensor as mt
import argparse
import time

"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()
"""
Hyper-parameters.
"""
num_epoch = args.num_epoch
lr = args.lr
batch_size = args.batch_size

"""
Generate training data.
"""
data = pd.read_csv("../data/Iris/Iris.csv").drop("Id", axis=1) # remove id column
data = data.sample(len(data), replace=False) # shuffle
# label encoding. translate string to int
le = LabelEncoder()
number_label = le.fit_transform(data["Species"]) # shape = (num_items,)
# one-hot encoding. translate int to ont-hot
oh = OneHotEncoder(sparse=False) # not in sparse matrix
one_hot_label = oh.fit_transform(number_label.reshape(-1, 1)) # shape = (num_items, num_labels)
# print(data)
train_set = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

"""
Construct computing graph.
"""
x = mt.core.Variable(dim=(4, 1), init=False, trainable=False)
label = mt.core.Variable(dim=(3, 1), init=False, trainable=False)

hidden1 = mt.layer.FC(x, 4, 20, "ReLU")
hidden2 = mt.layer.FC(hidden1, 20, 20, "ReLU")
output = mt.layer.FC(hidden2, 20, 3, "ReLU")

predict = mt.ops.SoftMax(output)
loss = mt.ops.loss.CrossEntropyWithSoftMax(output, label)
optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

"""
Training part.
"""
for epoch in range(num_epoch):
    batch_count = 0
    start_time = time.time()
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :]).T # shape = (4, 1)
        l = np.mat(one_hot_label[i, :]).T
        
        x.set_value(features)
        label.set_value(l)
        
        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()

        batch_count += 1
        if (batch_count == batch_size):
            optimizer.update()
            batch_count = 0

    end_time = time.time()
    pred = []
    for i in range(len(train_set)):
        feature = np.mat(train_set[i, :]).T
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.A.ravel())
    pred = np.array(pred).argmax(axis=1)
    accuracy = (number_label == pred).astype(np.int32).sum() / len(data)
    print("epoch: {:3d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))
