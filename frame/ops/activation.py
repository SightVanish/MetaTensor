import numpy as np
from .ops import Operator

# TODO
class Step(Operator):
    def compute(self):
        """
        Step function = 1 if x>0 else 0
        """
        self.value = np.mat(np.where(self.parents[0]>0.0, 1.0, 0.0))
    
    def get_jacobi(self, parent):
        np.mat(np.eye(self.dimension()))
        return super().get_jacobi(parent)

