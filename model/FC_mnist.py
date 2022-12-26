import sys
sys.path.append('..')
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import metatensor as mt
import time
import argparse
from LoadMnist import load_mnist_train
"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64)
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
# only take 5000 samples
X, y = load_mnist_train("../data/Mnist")
X, y = X[:5000] / 255, y.astype(np.int32)[:5000]
one_hot_label = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

"""
Generate training input.
"""
x = mt.core.Variable(dim=(784, 1), init=False, trainable=False)
label = mt.core.Variable(dim=(10, 1), init=False, trainable=False)
# hidden1 = mt.layer.FC(x, 784, 100, "ReLU")
# hidden2 = mt.layer.FC(hidden1, 100, 20, "ReLU")
# output = mt.layer.FC(hidden2, 20, 10, None)
output = mt.layer.FC(x, 784, 10, None)

predict = mt.ops.SoftMax(output)
loss = mt.ops.loss.CrossEntropyWithSoftMax(output, label)
optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

"""
Training part.
"""
for epoch in range(num_epoch):
    cur_batch_size = 0
    start_time = time.time()
    iter_start_time = time.time()
    for i in range(len(X)):
        features = np.mat(X[i]).T # shape = (4, 1)
        l = np.mat(one_hot_label[i]).T
        
        x.set_value(features)
        label.set_value(l)
        
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
    for i in range(len(X[:1000])):
        feature = np.mat(X[i]).T
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.A.ravel())
    pred = np.array(pred).argmax(axis=1)
    accuracy = (y[:1000] == pred).astype(np.int32).sum() / 1000
    print("epoch: {:d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))
