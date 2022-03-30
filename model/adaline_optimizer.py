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
x = mt.core.Variable(dim=(3, 1), init=False, trainable=False)
label = mt.core.Variable(dim=(1, 1), init=False, trainable=False)
w = mt.core.Variable(dim=(1, 3), init=True, trainable=True)
b = mt.core.Variable(dim=(1, 1), init=True, trainable=True)
output = mt.ops.Add(mt.ops.MatMul(w, x), b)
predict = mt.ops.Step(output)
loss = mt.ops.loss.PerceptionLoss(mt.ops.MatMul(label, output))

# define optimizer
optimizer = mt.optimizer.GradientDescent(mt.default_graph, loss, lr)

cur_batch_size = 0

"""
Training part, with optimizer.
"""
for epoch in range(num_epoch):
    start_time = time.time()
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        l = np.mat(train_set[i, -1])
        x.set_value(features)
        label.set_value(l)

        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()

        cur_batch_size += 1
        if (cur_batch_size == batch_size):
            optimizer.update()
            cur_batch_size = 0

    end_time = time.time()
    pred = []
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0, 0])
    pred = np.array(pred) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(np.int32).sum() / len(train_set)
    print("epoch: {:2d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))

"""
adaline.py uses a large node for variables. So adaline.py will be about 8x faster than adaline_optimizer.py.
"""