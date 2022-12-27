import sys
sys.path.append('..')
import numpy as np
import metatensor as mt
import time
import argparse
from sklearn.datasets import make_classification
"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k', type=int, default=20) # size of hidden vector
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
dimension = 60 # feature dimension
X, y = make_classification(600, dimension, n_informative=20) # only 20 dimensions are informative
y = y * 2 - 1

"""
Generate training input.
"""
x1 = mt.core.Variable(dim=(dimension, 1), init=False, trainable=False) # linear term
label = mt.core.Variable(dim=(1, 1), init=False, trainable=False) # label
w = mt.core.Variable(dim=(1, dimension), init=True, trainable=True) # linear weight
E = mt.core.Variable(dim=(k, dimension), init=True, trainable=True) # embedding vector
b = mt.core.Variable(dim=(1, 1), init=True, trainable=True) # bias

embedding = mt.ops.MatMul(E, x1)

# FM
fm = mt.ops.Add(
    mt.ops.MatMul(w, x1),
    mt.ops.MatMul(mt.ops.Reshape(embedding, shape=(1, k)), embedding)
)

# deep
hidden_1 = mt.layer.FC(embedding, k, 8, "ReLU") # (8, 1)
hidden_2 = mt.layer.FC(hidden_1, 8, 4, "ReLU") # (4, 1)
deep = mt.layer.FC(hidden_2, 4, 1, "ReLU") # (1, 1)

# output
output = mt.ops.Add(fm, deep, b)
predict = mt.ops.Logistic(output)
loss = mt.ops.LogLoss(mt.ops.Multiply(label, output))
optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

"""
Training part.
"""
for epoch in range(num_epoch):
    batch_count = 0
    start_time = time.time()
    iter_start_time = time.time()
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))
        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()
        batch_count += 1
        if (batch_count >= batch_size):
            iter_end_time = time.time()
            print("epoch: {:d}, {:.3f} sec/iter, iter: {:d}/{:d}, loss: {:3f}".format(epoch + 1, iter_end_time - iter_start_time, i + 1, len(X), loss.value[0, 0]))
            iter_start_time = time.time()

            optimizer.update()
            batch_count = 0

    end_time = time.time()
    pred = []
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        predict.forward()
        pred.append(predict.value[0, 0])
    pred = (np.array(pred) > 0.5).astype(np.int32) * 2 - 1
    accuracy = (y == pred).astype(np.int32).sum() / len(X)
    print("epoch: {:d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))


