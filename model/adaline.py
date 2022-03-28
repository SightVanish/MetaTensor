import sys
sys.path.append('..')
import numpy as np
import frame as mt
import time

"""
Generate training data.
"""
male_heights = np.random.normal(171, 6, 500) # create 500 samples, mean=171, std=6
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500

"""
Generate training input.
"""
# train_set[i].shape = (num_sample, feature dim + label dim)
train_set = np.array([np.concatenate((male_heights, female_heights)), 
                      np.concatenate((male_weights, female_weights)), 
                      np.concatenate((male_bfrs, female_bfrs)), 
                      np.concatenate((male_labels, female_labels)),]).T

np.random.shuffle(train_set)

"""
Construct computing graph.
"""
# input
x = mt.Variable(dim=(3,1), init=False, trainable=False)

# label
label = mt.Variable(dim=(1,1), init=False, trainable=False)

# weight
w = mt.Variable(dim=(1,3), init=True, trainable=True)

# bias
b = mt.Variable(dim=(1,1), init=True, trainable=True)

"""
Model part: y = step(w * x + b).
"""

output = mt.ops.Add(mt.ops.MatMul(w, x), b)
predict = mt.ops.Step(output)

# loss
loss = mt.ops.loss.PerceptionLoss(mt.ops.MatMul(label, output))

"""
Training part
"""
num_epoch = 100
lr = 1e-4 # learning rate

for epoch in range(num_epoch):
    start_time = time.time()
    for i in range(len(train_set)):
        feature = np.mat(train_set[i, :-1]).T # feature.shape = [feature dim, 1]
        l = np.mat(train_set[i, -1]).T # label.shape = [label dim, 1]

        x.set_value(feature)
        label.set_value(l)

        # forward
        loss.forward()
        # compute jacobi matrix
        w.backward(loss)
        b.backward(loss)

        # update value
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))

        # clear jacobi after updating trainable parameters
        mt.default_graph.clear_jacobi()
    end_time = time.time()
    # varify model precision for each epoch
    pred = []
    for i in range(len(train_set)):
        feature = np.mat(train_set[i, :-1]).T
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value[0, 0])
    # the output of model is 1/0 so we need to convert the result to 1/-1
    pred = np.array(pred) * 2 - 1

    accuracy = (train_set[:, -1] == pred).astype(np.int32).sum() / len(train_set)

    print("epoch: {:2d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))
