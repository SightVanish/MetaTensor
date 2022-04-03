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
        assert len(self.parents) == 1
        self.value = np.mat(np.where(self.parents[0].value > 0.0, 0.0, -self.parents[0].value))
    
    def get_jacobi(self, parent):
        diag = np.where(parent.value >= 0.0, 0.0, -1.0)
        return np.diag(diag.ravel())

class LogLoss(LossFunction):
    """
    Log loss.
    """
    def compute(self):
        assert len(self.parents) == 1
        x = self.parents[0].value
        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e3, 1e3, -x))) # np.where ensures the value of -x <= 1e3 (overflow)
    
    def get_jacobi(self, parent):
        x = self.parents[0].value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e3, 1e3, x)))
        return np.diag(diag.ravel())