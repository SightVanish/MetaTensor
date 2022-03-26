from itertools import Predicate
import sys
sys.path.append('../')
import frame as mt

import numpy as np

"""
generate training data
"""
male_heights = np.random.normal(171, 6, 500) # create 500 samples, mean=171, std=6
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [0] * 500

"""
generate training input
"""
train_set = np.array([
    np.concatenate((male_heights, female_heights)),
    np.concatenate((male_weights, female_weights)),
    np.concatenate((male_bfrs, female_bfrs)),
    np.concatenate((male_labels, female_labels)),
]).T # train_set[i] is a single training [feature, label]

np.random.shuffle(train_set)

"""
construct computing graph
"""
# input
x = mt.Variable(dim=(3,1), init=False, trainable=False)

# label
label = mt.Variable(dim=(1,1), init=False, trainable=False)

# weight
w = mt.Variable(dim=(1,3), init=True, trainable=True)

# bias
b = mt.Variable(dim=(1,1), init=True, trainable=True)

# output
output = mt.Add(mt.MatMul(w,x), b)
predict = mt.Step(output)

# loss
loss = mt.