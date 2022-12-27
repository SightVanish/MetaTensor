import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import metatensor as mt
import time
import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
"""
Parser.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k', type=int, default=2) # size of hidden vector
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
# read data
data = pd.read_csv('../data/Titanic/titanic.csv').drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# encode data
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False) # in spare matrix

Pclass = ohe.fit_transform(le.fit_transform(data['Pclass'].fillna(0)).reshape(-1, 1))
Sex = ohe.fit_transform(le.fit_transform(data['Sex'].fillna('')).reshape(-1, 1))
Embarked = ohe.fit_transform(le.fit_transform(data['Embarked'].fillna('')).reshape(-1, 1))

features = np.concatenate([
    Pclass,
    Sex,
    data[['Age']].fillna(0),
    data[['SibSp']].fillna(0),
    data[['Parch']].fillna(0),
    data[['Fare']].fillna(0),
    Embarked
], axis=1)

labels = data['Survived'].values * 2 - 1

dimension = features.shape[1]
"""
Generate training input.
"""
x = mt.core.Variable(dim=(dimension, 1), init=False, trainable=False) # linear term

# one-hot
x_Pclass = mt.core.Variable(dim=(Pclass.shape[1], 1), init=False, trainable=False)
x_Sex = mt.core.Variable(dim=(Sex.shape[1], 1), init=False, trainable=False)
x_Embarked = mt.core.Variable(dim=(Embarked.shape[1], 1), init=False, trainable=False)

label = mt.core.Variable(dim=(1, 1), init=False, trainable=False) # label

w = mt.core.Variable(dim=(1, dimension), init=True, trainable=True) # linear term weight

# embedding term
E_Pclass = mt.core.Variable(dim=(k, Pclass.shape[1]), init=True, trainable=True)
E_Sex = mt.core.Variable(dim=(k, Sex.shape[1]), init=True, trainable=True)
E_Embarked = mt.core.Variable(dim=(k, Embarked.shape[1]), init=True, trainable=True)

b = mt.core.Variable(dim=(1, 1), init=True, trainable=True) # bias

embedding = mt.ops.Concat(
    mt.ops.MatMul(E_Pclass, x_Pclass),
    mt.ops.MatMul(E_Sex, x_Sex),
    mt.ops.MatMul(E_Embarked, x_Embarked)
)

# FM
fm = mt.ops.Add(
    mt.ops.MatMul(w, x),
    mt.ops.MatMul(mt.ops.Reshape(embedding, shape=(1, 3 * k)), embedding)
)

# deep
hidden_1 = mt.layer.FC(embedding, 3 * k, 8, 'ReLU')
hidden_2 = mt.layer.FC(hidden_1, 8, 4, 'ReLU')
deep = mt.layer.FC(hidden_2, 4, 1, 'ReLU')

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
    for i in range(len(features)):
        x.set_value(np.mat(features[i]).T)
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)
        label.set_value(np.mat(labels[i]))
        # optimizr will be responsible for forward and backward propagation
        optimizer.one_step()
        batch_count += 1
        if (batch_count >= batch_size):
            iter_end_time = time.time()
            print("epoch: {:d}, {:.3f} sec/iter, iter: {:d}/{:d}, loss: {:3f}".format(epoch + 1, iter_end_time - iter_start_time, i + 1, len(features), loss.value[0, 0]))
            iter_start_time = time.time()

            optimizer.update()
            batch_count = 0

    end_time = time.time()
    pred = []
    for i in range(len(features)):
        x.set_value(np.mat(features[i]).T)
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)
        predict.forward()
        pred.append(predict.value[0, 0])
    pred = (np.array(pred) > 0.5).astype(np.int32) * 2 - 1
    accuracy = (labels == pred).astype(np.int32).sum() / len(features)
    print("epoch: {:d},  {:.3f} sec/epoch,  accuracy: {:.3f}".format(epoch + 1, end_time - start_time, accuracy))


