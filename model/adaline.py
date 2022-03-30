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
Hyper-parameters.
"""
num_epoch = 50
lr = 1e-4 # learning rate
batch_size = 10

"""
Construct computing graph.
"""
# input
X = mt.Variable(dim=(batch_size, 3), init=False, trainable=False)

# label
label = mt.Variable(dim=(batch_size, 1), init=False, trainable=False)

# weight
w = mt.Variable(dim=(3, 1), init=True, trainable=True)

# bias
b = mt.Variable(dim=(1, 1), init=True, trainable=True)

# to expand the dimension b
ones = mt.Variable(dim=(batch_size, 1), init=False, trainable=False)
ones.set_value(np.mat(np.ones(batch_size)).T) # shape = (batch_size,)

bias = mt.ops.ScalarMultiply(b, ones)

"""
Model part: y = step(w * x + b).
"""

output = mt.ops.Add(mt.ops.MatMul(X, w), bias)
predict = mt.ops.Step(output)

# loss
loss = mt.ops.loss.PerceptionLoss(mt.ops.Multiply(label, output))

# average loss value for a batch
B = mt.core.Variable(dim=(1, batch_size), init=False, trainable=False)
B.set_value(1 / batch_size * np.mat(np.ones(batch_size)))
mean_loss = mt.ops.MatMul(B, loss)
"""
Training part
"""

for epoch in range(num_epoch):
    start_time = time.time()
    for i in range(0, len(train_set), batch_size):
        feature = np.mat(train_set[i:i + batch_size, :-1]) # feature.shap = [batch_size, feature dim]
        l = np.mat(train_set[i:i + batch_size, -1]).T # label.shape = [batch_size, label dim]

        X.set_value(feature)
        label.set_value(l)

        # forward
        mean_loss.forward()

        # compute jacobi matrix
        w.backward(mean_loss)
        b.backward(mean_loss)

        # update value
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))

        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))
        # clear jacobi after updating trainable parameters
        mt.default_graph.clear_jacobi()
    
    end_time = time.time()
    # varify model precision for each epoch
    pred = []
    for i in range(0, len(train_set), batch_size):
        feature = np.mat(train_set[i:i+batch_size, :-1])
        X.set_value(feature)
        predict.forward()
        pred.extend(predict.value.A.ravel()) # it must be extend, not append
    # the output of model is 1/0 so we need to convert the result to 1/-1
    pred = np.array(pred) * 2 - 1

    accuracy = (train_set[:, -1] == pred).astype(np.int32).sum() / len(train_set)

    print("epoch: {:2d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))
