import numpy as np
from ..computational_graph.node import Node

def fill_diagonal(target, filler):
    """
    This is a helper function fo MatMul.get_jacobi()
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
    Base class, Operator is the same with Node
    """
    pass

class MatMul(Operator):
    """
    Matrix multiply
    C = A * B
    A.shape = (m, n), B.shape = (n, k), C.shape = (m, k)
    """
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[0].value # self.value is numpy.array

    def get_jacobi(self, parent):
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]: 
            # Jacobi(C/A).shape = (mk, mn)
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            # Jacobi(C/B).shape = (mk, nk)
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimensions()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimensions()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]

class Add(Operator):
    def compute(self):
        """
        Matrix add
        """
        self.value = np.mat(np.zeros(self.parent[0].shape))
        for parent in self.parents:
            self.value += parent.value
    
    def get_jacobi(self, parent):
        # jacobi is identiy matrix for all parents
        return np.mat(np.eye(self.dimension()))

