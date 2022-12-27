"""
Accept inputs with different sequence sizes.
"""
import sys
sys.path.append('..')
import numpy as np
import metatensor as mt
from scipy import signal
import argparse
import time

def get_sequence_data(dimension=10, length=20, num_examples=1000, train_set_ratio=0.7, seed=42):
    """
    Construct dataset with sin and square samples.
    """
    xx = []
    # sin
    xx.append(np.sin(np.arange(0, 10, 10/length)).reshape(-1, 1))
    # square
    xx.append(np.array(signal.square(np.arange(0, 10, 10/length))).reshape(-1, 1))
    data = []
    for i in range(2):
        x = xx[i]
        for j in range(num_examples // 2):
            sequence = x + np.random.normal(0, 0.6, (len(x), dimension)) # add noise
            label = np.array([int(i == k) for k in range(2)]) # (1, 0) or (0, 1)
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)

    train_set_size = int(num_examples * train_set_ratio)
    return (
        data[:train_set_size, :-2].reshape(-1, length, dimension), # train set features
        data[:train_set_size, -2:], # train set labels
        data[train_set_size:, :-2].reshape(-1, length, dimension), # valid set features
        data[train_set_size:, -2:]
    )

"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=96, help='sequence length')
parser.add_argument('--dimension', type=int, default=16, help='input dimension')
parser.add_argument('--status_dimension', type=int, default=12)
args = parser.parse_args()

"""
Hyper-parameters.
"""
num_epoch = args.num_epoch
lr = args.lr
batch_size = args.batch_size
seq_len = args.seq_len
dimension = args.dimension
status_dimension = args.status_dimension

"""
Generate training data.
"""
signal_train, label_train, signal_test, label_test = get_sequence_data(length=seq_len, dimension=dimension)

inputs = [mt.core.Variable(dim=(dimension, 1), init=False, trainable=False) for _ in range(seq_len)] # input nodes
label = mt.core.Variable(dim=(2, 1), init=False, trainable=False)
U = mt.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True) # hidden status matrix
W = mt.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True) # status weight matrix
b = mt.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

hiddens = [] # store all hidden nodes
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
    hiddens.append(last_status)

welding_point = mt.ops.Welding() # welding point

fc1 = mt.layer.FC(welding_point, status_dimension, 40, 'ReLU')
fc2 = mt.layer.FC(fc1, 40, 10, 'ReLU')

# output
output = mt.layer.FC(fc2, 10, 2, 'None')
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
        # intercept input vector
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start: end]

        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)
        welding_point.weld(hiddens[j]) # weld to the last hidden node

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


