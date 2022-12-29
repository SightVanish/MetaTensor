import sys
sys.path.append('..')
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from LoadMnist import load_mnist_train
import metatensor as mt
import time
import argparse
"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32)
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
X, y = load_mnist_train("../data/Mnist")
X, y = X[:5000] / 255, y.astype(np.int32)[:5000]
one_hot_label = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
img_shape = (28, 28)
"""
Generate training input.
"""
x = mt.core.Variable(dim=img_shape, init=False, trainable=False)
label = mt.core.Variable(dim=(10, 1), init=False, trainable=False)

conv1 = mt.layer.Conv([x], img_shape, 3, (5, 5), 'ReLU')
pooling1 = mt.layer.Pooling(conv1, (3, 3), (2, 2))
conv2 = mt.layer.Conv(pooling1, (14, 14), 3, (3, 3), 'ReLU')
pooling2 = mt.layer.Pooling(conv2, (3, 3), (2, 2))

fc1 = mt.layer.FC(mt.ops.Concat(*pooling2), 147, 120, 'ReLU')

output = mt.layer.FC(fc1, 120, 10, 'None')

predict = mt.ops.SoftMax(output)
loss = mt.ops.loss.CrossEntropyWithSoftMax(output, label)
optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

"""
Training part.
"""
for epoch in range(num_epoch):
    batch_count = 0
    start_time = time.time()
    iter_start_time = time.time()
    for i in range(len(X)):
        features = np.mat(X[i]).reshape(img_shape)
        l = np.mat(one_hot_label[i]).T
        
        x.set_value(features)
        label.set_value(l)
        
        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()

        batch_count += 1
        if (batch_count == batch_size):
            iter_end_time = time.time()
            print("epoch: {:d}, {:.3f} sec/iter, iter: {:d}/{:d}, loss: {:3f}".format(epoch + 1, iter_end_time - iter_start_time, i + 1, len(X), loss.value[0, 0]))
            iter_start_time = time.time()

            optimizer.update()
            batch_count = 0

    end_time = time.time()
    pred = []
    for i in range(len(X[:1000])):
        feature = np.mat(X[i]).reshape(img_shape)
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.A.ravel())
    pred = np.array(pred).argmax(axis=1)
    accuracy = (y[:1000] == pred).astype(np.int32).sum() / 1000
    print("epoch: {:d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))
