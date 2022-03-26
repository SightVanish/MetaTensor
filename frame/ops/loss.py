import numpy as np
from ..computational_graph import Node

class LossFunction(Node):
    """
    Base class, LossFunction is the same with Node
    """
    pass

class PerceptionLoss(LossFunction):
    """
    
    """
    def compute(self):
        self.value = 