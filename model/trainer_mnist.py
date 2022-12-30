import sys
sys.path.append('..')
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import metatensor as mt
import time
import argparse
from LoadMnist import load_mnist_train
# """
# Parser.
# """
# parser = argparse.ArgumentParser()
# parser.add_argument('--num_epoch', type=int, default=8)
# parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
# parser.add_argument('--batch_size', type=int, default=64)
# args = parser.parse_args()
# """
# Hyper-parameters.
# """
# num_epoch = args.num_epoch
# lr = args.lr
# batch_size = args.batch_size

# """
# Generate training data.
# """
# # only take 5000 samples
# X, y = load_mnist_train("../data/Mnist")
# X, y = X[:1000] / 255, y.astype(np.int32)[:1000]
# one_hot_label = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

# """
# Generate training input.
# """
# x = mt.core.Variable(dim=(784, 1), init=False, trainable=False, name='input_node')
# label = mt.core.Variable(dim=(10, 1), init=False, trainable=False)
# # hidden1 = mt.layer.FC(x, 784, 100, "ReLU")
# # hidden2 = mt.layer.FC(hidden1, 100, 20, "ReLU")
# # output = mt.layer.FC(hidden2, 20, 10, None)
# output = mt.layer.FC(x, 784, 10, None)

# predict = mt.ops.SoftMax(output, name='predict')
# loss = mt.ops.loss.CrossEntropyWithSoftMax(output, label)
# optimizer = mt.optimizer.Adam(mt.default_graph, loss, lr)

# """
# Training part.
# """
# accuracy = mt.ops.Accuray(output, label)
# precision = mt.ops.Precision(output, label)
# recall = mt.ops.Recall(output, label)

# feature = x
# trainer = mt.trainer.SimpleTrainer(
#     [feature], label, loss, optimizer, num_epoch, batch_size, eval_on_train=True, metrics_ops=[accuracy, precision, recall])


# trainer.train_and_eval({feature.name: X}, one_hot_label, {feature.name: X}, one_hot_label)

# saver = mt.trainer.Saver('./save')

# saver.save(model_file_name='model.json', weights_file_name='weights.npz')





saver = mt.trainer.Saver('./save')
saver.load(model_file_name='model.json', weights_file_name='weights.npz')

X, y = load_mnist_train("../data/Mnist")
X, y = X[:100] / 255, y.astype(np.int32)[:100]

x = mt.get_node_from_graph('input_node')
predict = mt.get_node_from_graph('predict')

for i in range(len(X)):
    x.set_value(np.mat(X[i]).T)

    predict.forward()
    ground_truth = y[i]
    print('model predict {} and ground truth: {}'.format(np.argmax(predict), ground_truth))