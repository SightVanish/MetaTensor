import os
import struct
import numpy as np
import gzip
import sys
def load_mnist_train(path):
    labels_path = os.path.join(path,'train-labels-idx1-ubyte.gz')
    images_path = os.path.join(path,'train-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromstring(lbpath.read(),dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromstring(imgpath.read(),dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
