import abc
from unicodedata import name # Abstract base classes
import numpy as np
from .graph import default_graph

class Node(object):
    """
    Base class for nodes in computing graph
    """
    def __init__(self, *parents, **kargs):
        # set parameters
        self.kargs = kargs
         # if 'graph' is not found in kargs, self.graph = default_graph
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        self.parents = list(parents)
        self.children = [] # init empty children list
        self.value = None # value of this node
        self.jacobi = None # jacobi matrix of result node to this node

        # add children list backwards
        for parent in self.parents:
            parent.children.append(self)
        
        # add this node to compute graph
        self.graph.add_node(self)

    def get_parents(self):
        """
        Return all parent nodes.
        """
        return self.parents
    
    def get_children(self):
        """
        Return all children nodes.
        """
        return self.children
    
    def gen_node_name(self, **kargs):
        """
        Renerate node name.
        """
        # If the node name is not specified in kargs, set default node name like 'MatMul:3'
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()
        ))
        # add name_scope, like 'DNN/MatMul:3'
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)
    
    def forward(self):
        """
        Forward computation.
        """
        # recursively call forward()
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()

    @abc.abstractmethod
    def compute(self):
        """        
        Compute the value of this node based on its parent nodes.
        """

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        Compute Jacobi of this node to each parent node.
        """
    
    def backward(self, result):
        """
        Backward computation. Compute Jacobi of result node to this node.
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension())) # jacobi is identity matrix
            else:
                self.jacobi = np.mat(np.zeros((result.dimension(), self.dimension())))
                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * child.get_jacobi(self)
        return self.jacobi

    def clear_jacobi(self):
        """
        Clear jacobi matrix of this node.
        """
        self.jacobi = None

    def dimension(self):
        """
        Return the size of this node.
        """
        # we only consider 2-D matrix
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
        Return the shape of this node.
        """
        return self.value.shape

    def reset_value(self, recursive=True):
        """
        Reset the values of this node and all its descendants.
        """
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):
    """
    Variable node
    """
    def __init__(self, dim, init=False, trainable=True, **kargs):
        """
        Set the value of this variable randomly if init==True.
        """
        # dim is a tuple, like (height, width)
        super(Variable, self).__init__(**kargs)
        self.dim = dim
        self.trainable = trainable

        if init:
            # init with normal distribution
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

    def set_value(self, value):
        """
        Set the value of this variable to given value.
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # the value of this node is changed so all its decendants should be modified
        self.reset_value()
        self.value = value

class name_scope(object):
    def __init__(self, name_scope):
        self.name_scope = name_scope

    def __enter__(self):
        default_graph.name_scope = self.name_scope
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        default_graph.name_scope = None