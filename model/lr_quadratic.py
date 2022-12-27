"""
Construct quadratic features manually.
"""
import sys
sys.path.append('..')
import numpy as np
import metatensor as mt
from sklearn.datasets import make_circles
import argparse
import time

"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--use_quadratic', action="store_true", help="add quadratic term")
args = parser.parse_args()
"""
Hyper-parameters.
"""
num_epoch = args.num_epoch
lr = args.lr
batch_size = args.batch_size
use_quadratic = args.use_quadratic
print('Quadratic: ', use_quadratic)
# get concentric distribution data
X, y = make_circles(200, noise=0.2, factor=0.4)
y = y * 2 - 1 # from 0/1 -> -1/1

"""
Construct computing graph.
"""
x1 = mt.core.Variable(dim=(2,1), init=False, trainable=False) # first order input, original input
label = mt.core.Variable(dim=(1,1), init=False, trainable=False)
b = mt.core.Variable(dim=(1,1), init=True, trainable=True)

if use_quadratic:
    # x2 = mt.ops.Reshape(
    #     mt.ops.MatMul(x1, mt.ops.Reshape(x1, shape=(1,2))),
    #     shape=(4,1)) # x2 = x1 * x1.T
    # x = mt.ops.Concat(x1, x2) # x.shape = (6, 1)
    # w = mt.core.Variable(dim=(1,6), init=True, trainable=True)
    # output = mt.ops.Add(mt.ops.MatMul(w, x), b) # w * x + b

    # quadratic in compact form: x1*w + x1.T*W*x1 + b
    w = mt.core.Variable(dim=(1, 2), init=True, trainable=True)
    W = mt.core.Variable(dim=(2, 2), init=True, trainable=True)
    b = mt.core.Variable(dim=(1, 1), init=True, trainable=True)

    output = mt.ops.Add(
        mt.ops.MatMul(w, x1), 
        mt.ops.MatMul(mt.ops.Reshape(x1, shape=(1, 2)), mt.ops.MatMul(W, x1)),
        b)
else:
    x = x1
    w = mt.core.Variable(dim=(1,2), init=True, trainable=True)
    output = mt.ops.Add(mt.ops.MatMul(w, x), b) # w * x + b



predict = mt.ops.Logistic(output)

loss = mt.ops.loss.LogLoss(mt.ops.Multiply(label, output))
optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

"""
Training part.
"""
for epoch in range(num_epoch):
    batch_count = 0
    start_time = time.time()
    for i in range(len(X)):
        features = np.mat(X[i]).T # shape = (4, 1)
        l = np.mat(y[i])
        
        x1.set_value(features)
        label.set_value(l)
        
        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()

        batch_count += 1
        if (batch_count == batch_size):
            optimizer.update()
            batch_count = 0

    end_time = time.time()
    pred = []
    for i in range(len(X)):
        feature = np.mat(X[i]).T
        x1.set_value(feature)
        predict.forward()
        pred.append(predict.value[0, 0])
    pred = (np.array(pred) > 0.5).astype(np.int32) * 2 -1
    accuracy = (y == pred).astype(np.int32).sum() / len(X)
    print("epoch: {:2d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))

