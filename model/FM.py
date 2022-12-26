import sys
sys.path.append('..')
import numpy as np
import metatensor as mt
import time
import argparse
from sklearn.datasets import make_circles
"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k', type=int, default=2) # hidden features
args = parser.parse_args()
"""
Hyper-parameters.
"""
num_epoch = args.num_epoch
lr = args.lr
batch_size = args.batch_size
k = args.k
"""
Generate training data.
"""
X, y = make_circles(600, noise=0.1, factor=0.2)
y = y * 2 - 1
dimension = 20 # feature dimension
# add noise features
X = np.concatenate([X, np.random.normal(0.0, 0.01, (600, dimension-2))], axis=1)
"""
Generate training input.
"""
x1 = mt.core.Variable(dim=(dimension, 1), init=False, trainable=False) # linear term
label = mt.core.Variable(dim=(1, 1), init=False, trainable=False) # label
w = mt.core.Variable(dim=(1, dimension), init=True, trainable=True, name='w') # linear weight
H = mt.core.Variable(dim=(k, dimension), init=True, trainable=True, name='H') # hidden weight
HTH = mt.ops.MatMul(mt.ops.Reshape(H, shape=(dimension, k)), H) # H^T @ H
b = mt.core.Variable(dim=(1, 1), init=True, trainable=True, name='x') # bias

# w*x1 + x1^T*H^T*H*x1 + b
output = mt.ops.Add(
    # linear term
    mt.ops.MatMul(w, x1), 
    # quadratic term
    mt.ops.MatMul(
        mt.ops.Reshape(x1, shape=(1, dimension)),
        mt.ops.MatMul(HTH, x1)),
    b)

predict = mt.ops.Logistic(output)
loss = mt.ops.LogLoss(mt.ops.Multiply(label, output))
optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

"""
Training part.
"""
for epoch in range(num_epoch):
    cur_batch_size = 0
    start_time = time.time()
    iter_start_time = time.time()
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))
        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()
        cur_batch_size += 1
        if (cur_batch_size == batch_size):
            iter_end_time = time.time()
            print("epoch: {:d}, {:.3f} sec/iter, iter: {:d}/{:d}, loss: {:3f}".format(epoch + 1, iter_end_time - iter_start_time, i + 1, len(X), loss.value[0, 0]))
            iter_start_time = time.time()

            optimizer.update()
            cur_batch_size = 0

    end_time = time.time()
    pred = []
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        predict.forward()
        pred.append(predict.value[0, 0])
    pred = (np.array(pred) > 0.5).astype(np.int32) * 2 - 1
    accuracy = (y == pred).astype(np.int32).sum() / len(X)
    print("epoch: {:d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))


