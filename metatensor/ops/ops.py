from webbrowser import Opera
from joblib import parallel_backend
import numpy as np
from ..core.node import Node

def fill_diagonal(target, filler):
    """
    Helper function for MatMul.get_jacobi()
    """
    # C = A * B
    # A.shape = (m, n), B.shape = (n, k), C.shape = (m, k)
    # traget.shape = Jacobi.shape = (mk, mn) or (mk, nk)
    # filler.shape = B^T.shape = (k, n) or filler.shape = A.shape = (m,n)

    assert target.shape[0] / filler.shape[0] == target.shape[1] / filler.shape[1]

    r, c = filler.shape[0], filler.shape[1]
    for i in range(int(target.shape[0] / filler.shape[0])):
        target[i * r : (i + 1) * r, i * c : (i + 1) * c] = filler
    return target

class Operator(Node):
    """
    Abstract class for operators.
    """
    pass

class MatMul(Operator):
    """
    Matrix product.
    """
    # C = A * B
    # A.shape = (m, n), B.shape = (n, k), C.shape = (m, k)
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value # self.value is numpy.array

    def get_jacobi(self, parent):
        assert parent in self.parents
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]: 
            # Jacobi(C/A).shape = (mk, mn)
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            # Jacobi(C/B).shape = (mk, nk)
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]

class ScalarMultiply(Operator):
    """
    Matrix product. Parent[0].shape = 1x1.
    """
    def compute(self):
        assert self.parents[0].shape() == (1, 1) # make sure the first variable is a scalar
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)
    
    def get_jacobi(self, parent):
        assert parent in self.parents
        if parent is self.parents[0]:
            # shape = (parent[1].dimension, 1)
            return self.parents[1].value.flatten().T # flatten() will allocate a new memory space.
        else:
            # shape = (parent[1].dimension, parent[1].dimension)
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]

class Multiply(Operator):
    """
    Matrix dot product.
    """
    def compute(self):
        assert self.parents[0].shape() == self.parents[1].shape()
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        assert parent in self.parents
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A.ravel()) # .A1 = .A.ravel()
        else:
            return np.diag(self.parents[0].value.A.ravel())

class Add(Operator):
    def compute(self):
        """
        Matrix addition.
        """
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        for parent in self.parents:
            self.value += parent.value
    
    def get_jacobi(self, parent):
        assert parent in self.parents
        # jacobi is identiy matrix for all parents
        return np.mat(np.eye(self.dimension()))

# activation function
class Step(Operator):
    """
    Step function. y = 1 if x > 0, y = 0 if x<=0.
    """
    def compute(self):
        assert len(self.parents) == 1
        self.value = np.mat(np.where(self.parents[0].value > 0.0, 1.0, 0.0))
    
    def get_jacobi(self, parent):
        assert parent in self.parents
        np.mat(np.eye(self.dimension()))
        return super().get_jacobi(parent)

class Logistic(Operator):
    """
    Logistic function. y = 1 / (1 + e^(-x)).
    """
    def compute(self):
        assert len(self.parents) == 1
        x = self.parents[0].value
        self.value = np.mat(1.0 / (1.0 + np.power(np.e, np.where(-x > 1e3, 1e3, -x))))
    
    def get_jacobi(self, parent):
        # jacobi = (e^(-x) / (1 + e^(-x))^2) = Logistic * (1 - Logistic).
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)

class SoftMax(Operator):
    """
    Softmax function.
    """
    # this method can be called without instantiating this class.
    @staticmethod
    def softmax(a):
        # a is numpy array/matrix
        a[a > 1e3] = 1e3 # you have to change the value of a
        ep = np.power(np.e, a)
        return ep / np.sum(ep)
    
    def compute(self):
        assert len(self.parents) == 1
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        print("SoftMax node is only used for prediction. Do not use its get_jacobi.")
        raise NotImplementedError

class ReLU(Operator):
    """
    Rectified Linear Unit. This is actually leaky-ReLU.
    """
    nslope = 0.1 # slope of the negative x-axis
    def compute(self):
        assert len(self.parents) == 1
        self.value = np.mat(
            np.where(self.parents[0].value > 0.0, 
                     self.parents[0].value, 
                     self.nslope * self.parents[0].value))
    
    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, self.nslope))

# tensor operator
class Reshape(Operator):
    """
    Change the shape of matrix. Parameters: shape should be a tuple.
    """
    def __init__(self, *parents, **kargs):
        super().__init__(*parents, **kargs)
        
        self.new_shape = kargs.get('shape')
        assert isinstance(self.new_shape, tuple) and len(self.new_shape) == 2
    
    def compute(self):
        # change shape in numpy
        self.value = self.parents[0].value.reshape(self.new_shape)

    def get_jacobi(self, parent):
        # jacobi is identity matrix
        assert parent is self.parents[0]
        assert len(self.parents) == 1
        return np.mat(np.eye(self.dimension()))

class Concat(Operator):
    """
    Concatenate nodes to one node. Unfold all nodes in parents via row and concatenate to a (n, 1) node.
    """
    def compute(self):
        assert len(self.parents) > 1
        # axis=1 only works for np.mat
        self.value = np.concatenate([p.value.flatten() for p in self.parents], axis=1).T # shape = (num_parents*len_parent[0])

    def get_jacobi(self, parent):
        assert parent in self.parents
        dims = [p.dimension() for p in self.parents]
        index = self.parents.index(parent)
        dim = parent.dimension()

        assert dim == dims[index]
        jacobi = np.mat(np.zeros((self.dimension(), dim)))
        
        # only part of jacobi will be assigned as identity matrix
        start_row = int(np.sum(dims[:index]))
        end_row = start_row + dim
        jacobi[start_row:end_row, :] = np.eye(dim)
        return jacobi

class Welding(Operator):
    """
    This node can be welded to any node.
    """
    def compute(self):
        assert len(self.parents) == 1 and self.parents[0] is not None
        self.value = self.parents[0].value
    
    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))
    
    def weld(self, node):
        """
        Weld this node to input node.
        """
        # unweld
        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)
        self.parents.clear()
        # weld
        self.parents.append(node)
        node.children.append(self)

class Convolve(Operator):
    """
    2-D convolution. parents[0] * parents[1]
    """
    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)

        self.padded = None
    
    def compute(self):
        data = self.parents[0].value
        kernel = self.parents[1].value
        # weight, height
        w, h = data.shape
        kw, kh = kernel.shape
        hkw, hkh = kw // 2, kh // 2 # half of kernel size

        # padding
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2))) # padded image size
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw+w, hkh:hkh+h] = data
        
        self.value = np.mat(np.zeros((w, h)))

        # convolution
        for i in range(hkw, hkw + w):
            for j in range(hkh, hkh + h):
                self.value[i-hkw, j-hkh] = np.sum(np.multiply(self.padded[i-hkw:i-hkw+kw, j-hkh:j-hkh+kh], kernel))

    def get_jacobi(self, parent):
        assert parent in self.parents
        data = self.parents[0].value
        kernel = self.parents[1].value

        # weight, height
        w, h = data.shape
        kw, kh = kernel.shape
        hkw, hkh = kw // 2, kh // 2 # half of kernel size

        # padding
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2))) # padded image size

        # jacobi
        jacobi = []
        if parent is self.parents[0]:
            for i in range(hkw, hkw + w):
                for j in range(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i-hkw:i-hkw+kw, j-hkh:j-hkh+kh] = kernel
                    jacobi.append(mask[hkw:hkw+w, hkh:hkh+h].A1)
        elif parent is self.parents[1]:
            for i in range(hkw, hkw + w):
                for j in range(hkh, hkh + h):
                    jacobi.append(self.padded[i-hkw:i-hkw+kw, j-hkh:j-hkh+kh].A1)

        return np.mat(jacobi)

class MaxPooling(Operator):
    """
    Max-pooling.
    """
    def __init__(self, *parents, **kargs):
        """
        # Parameters:
        stride: kernel sliding stride
        size: size of kernel
        """
        Operator.__init__(self, *parents, **kargs)

        self.stride = kargs.get('stride')
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None # record which elements are selected in max pooling
    
    def compute(self):
        data = self.parents[0].value # feature maps
        w, h = data.shape
        dim = w * h
        sw, sh = self.stride
        hkw, hkh = sw // 2, sh // 2

        result = []
        flag = []
        for i in range(0, w, sw):
            row = [] # max values in this row
            for j in range(0, h, sh):
                # check pooling window boundary
                top, bottom = max(0, i-hkw), min(w, i+hkw+1)
                left, right = max(0, j-hkh), min(h, j+hkh+1)
                window = data[top:bottom, left:right]
                row.append(np.max(window))

                # record the position of max value in pooling window
                pos = np.argmax(window)
                w_width = right - left
                offset = (top + pos // w_width) * w + (left + pos % w_width)
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)
            result.append(row)
        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):
        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag
                