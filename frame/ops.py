from node import Node
import numpy as np

def fill_diagonal(target, filler):
    """
    target is zero matrix
    set target's diagonal value to filler
    """
    assert target.shape[0] / filler.shape[0] == target.shape[1] / filler.shape[1]

    r, c = filler.shape[0], filler.shape[1]
    for i in range(int(target.shape[0] / filler.shape[0])):
        target[i * r : (i + 1) * r, i * c : (i + 1) * c] = filler
    return target


"""
Base clasee
Operation is same with Node
"""
class Operator(Node):
    pass

"""
Matrix multiply
"""
class MatMul(Operator):
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]

        self.value = self.parents[0].value * self.parents[0].value

    def get_jacobi(self, parent):
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimensions()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimensions()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]

"""
Matrix add
"""
class Add(Operator):
    def compute(self):
        self.value = np.mat(np.zeros(self.parent[0].shape))
        for parent in self.parents:
            self.value += parent.value
    
    def get_jacobi(self, parent):
        # jacobi is identiy matrix for all parents
        return np.mat(np.eye(self.dimension()))




