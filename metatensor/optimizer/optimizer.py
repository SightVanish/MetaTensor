import abc
import numpy as np
from ..core.node import Node, Variable
from ..core.graph import Graph
from ..core.utils import get_node_from_graph

class Optimizer(Node):
    """
    Base class for optimizers. target is the node we want to optimize.
    """
    def __init__(self, graph, target, learning_rate):
        assert isinstance(target, Node) and isinstance(graph, Graph)
        assert learning_rate != 0
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        self.acc_gradient = dict() # accumulate gradient for a batch
        self.acc_no = 0

    def one_step(self):
        """
        Compute and accumulate gradient.
        """
        self.forward_backward()
        self.acc_no += 1
    
    def get_gradient(self, node):
        """
        Return the average gradient of a node.
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self):
        """
        Update value based on gradient.
        """
    
    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape

                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                self.acc_no = 1
            else:
                self.acc_no = acc_no
    
    def update(self, var_gradient=None):
        if var_gradient is not None:
            self.apply_gradients(var_gradient)
        # update parameters and clear gradient
        self._update()
        self.acc_gradient.clear()
        self.acc_no = 0

    def forward_backward(self):
        """
        Forward propagation and backward propagation. Optimizer is only available for loss node (a scalar).
        """
        # forward
        self.graph.clear_jacobi()
        self.target.forward()

        # backward
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)
                # the final result is a scalar, so we reshape it
                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient

class GradientDescent(Optimizer):
    """
    Gradient descenting optimizer.
    """
    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                # update node value
                node.set_value(node.value - self.learning_rate * gradient)

class Momentum(Optimizer):
    """
    Momentum optimizer. Take the old gradient into account.
    """
    def __init__(self, graph, target, learning_rate, momentum=0.9):
        super().__init__(graph, target, learning_rate)
        assert 0.0 < momentum < 1.0
        self.momentum = momentum
        self.v = {} # old gradient
        for node in self.graph.nodes:
            self.v[node] = 0

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                # update value
                self.v[node] = self.momentum * self.v[node] \
                        - self.learning_rate * gradient
                node.set_value(node.value + self.v[node])

class AdaGrad(Optimizer):
    """
    AdaGrad optimizer. Impose different learning rate for different dimension of node.
    """
    def __init__(self, graph, target, learning_rate):
        super().__init__(graph, target, learning_rate)
        # self.s[node] = gradient**2
        self.s = {}
        for node in self.graph.nodes:
            self.s[node] = 0

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                self.s[node] += np.power(gradient, 2)
                node.set_value(node.value - self.learning_rate * gradient / (np.sqrt(self.s[node] + 1e-10)))

class RMSProp(Optimizer):
    """
    Root Mean Square Propagation optimizer. Introduce locality to AdaGrad.
    """
    def __init__(self, graph, target, learning_rate, beta=0.9):
        super().__init__(graph, target, learning_rate)
        assert 0.0 < beta < 1.0
        self.beta = beta
        self.s = {}
        for node in self.graph.nodes:
            self.s[node] = 0
    
    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                # update value
                self.s[node] = self.beta * self.s[node] \
                            + (1 - self.beta) * np.power(gradient, 2)
                node.set_value(node.value \
                            - self.learning_rate * gradient / (np.sqrt(self.s[node] + 1e-10)))

class Adam(Optimizer):
    """
    Adaptive Momentum Estimation optimizer. Momentum + RMSProp.
    """
    def __init__(self, graph, target, learning_rate, momentum=0.9, beta=0.99):
        super().__init__(graph, target, learning_rate)
        assert 0.0 < momentum < 1.0
        assert 0.0 < beta < 1.0
        self.momentum = momentum
        self.beta = beta
        self.v = {}
        self.s = {}
        for node in self.graph.nodes:
            self.v[node] = 0
            self.s[node] = 0
    
    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                # update value
                self.v[node] = self.momentum * self.v[node] \
                        + (1 - self.momentum) * gradient
                self.s[node] = self.beta * self.s[node] \
                        + (1 - self.beta) * np.power(gradient, 2)
                node.set_value(node.value - self.learning_rate * self.v[node] / np.sqrt(self.s[node] + 1e-10))

                self.v[node] = self.momentum * self.v[node] + \
                    (1 - self.momentum) * gradient
