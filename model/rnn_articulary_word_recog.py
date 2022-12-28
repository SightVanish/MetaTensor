import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import metatensor as mt
import argparse
import time

"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--status_dimension', type=int, default=32)
args = parser.parse_args()

"""
Hyper-parameters.
"""
num_epoch = args.num_epoch
lr = args.lr
batch_size = args.batch_size
status_dimension = args.status_dimension

"""
Generate dataset.
"""
path_train = '../data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.arff'
path_test = '../data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.arff'

train, test = arff.loadarff(path_train), arff.loadarff(path_test)
train, test = pd.DataFrame(train[0]), pd.DataFrame(test[0])
signal_train = np.array([np.array([list(channel) for channel in sample]).T for sample in train['relationalAtt']])
signal_test = np.array([np.array([list(channel) for channel in sample]).T for sample in test['relationalAtt']])

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

label_train = ohe.fit_transform(le.fit_transform(train['classAttribute']).reshape(-1, 1))
label_test = ohe.fit_transform(le.fit_transform(test['classAttribute']).reshape(-1, 1))

seq_len = signal_train.shape[1]
dimension = signal_train.shape[2]

inputs = [mt.core.Variable(dim=(dimension, 1), init=False, trainable=False) for _ in range(seq_len)] # input nodes
label = mt.core.Variable(dim=(label_train.shape[1], 1), init=False, trainable=False)
U = mt.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True) # hidden status matrix
W = mt.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True) # status weight matrix
b = mt.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

last_status = None
for iv in inputs:
    h = mt.ops.Add(mt.ops.MatMul(U, iv), b) # hidden status
    if last_status:
        h = mt.ops.Add(
            mt.ops.MatMul(W, last_status),
            h
        )
    h = mt.ops.ReLU(h)

    last_status = h

fc1 = mt.layer.FC(h, status_dimension, 64, 'ReLU')
fc2 = mt.layer.FC(fc1, 64, 32, 'ReLU')

# output
output = mt.layer.FC(fc2, 32, 25, 'None')
predict = mt.ops.Logistic(output)
loss = mt.ops.CrossEntropyWithSoftMax(output, label)
optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

"""
Training part.
"""
for epoch in range(num_epoch):
    batch_count = 0
    start_time = time.time()
    iter_start_time = time.time()
    for i, s in enumerate(signal_train):
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        label.set_value(np.mat(label_train[i, :]).T)
        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()
        batch_count += 1
        if (batch_count >= batch_size):
            iter_end_time = time.time()
            print("epoch: {:d}, {:.3f} sec/iter, iter: {:d}/{:d}, loss: {:3f}".format(epoch + 1, iter_end_time - iter_start_time, i + 1, len(signal_train), loss.value[0, 0]))
            iter_start_time = time.time()

            optimizer.update()
            batch_count = 0

    end_time = time.time()
    pred = []
    for i, s in enumerate(signal_test):
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)
        predict.forward()
        pred.append(predict.value.A.ravel())
    pred = np.array(pred).argmax(axis=1)
    y = label_test.argmax(axis=1)
    accuracy = (y == pred).astype(np.int32).sum() / len(signal_test)
    print("epoch: {:d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))


