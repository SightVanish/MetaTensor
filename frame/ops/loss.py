import numpy as np
from ..core import Node

class LossFunction(Node):
    """
    Base class, same with Node.
    """
    pass

class PerceptionLoss(LossFunction):
    """
    Simplest loss for perception machine. y = 0 if x > 0, y = -x if x < 0.
    """
    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value > 0.0, 0.0, -self.parents[0].value))
    
    def get_jacobi(self, parent):
        diag = np.where(parent.value >= 0.0, 0.0, -1.0)
        return np.diag(diag.ravel())
