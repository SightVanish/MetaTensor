import abc
import numpy as np
from graph import default_graph

class Node(object):
    """
    Base class for nodes in computing graph
    """
    def __init__(self, *parents, **kargs):
        """
        init node
        """
        # set parameters
        self.graph = kargs.get('graph', default_graph) # if 'graph' is not found in kargs, self.graph = default_graph
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        self.parents = list(parents)
        self.children = [] # init empty children list
        self.value = None
        self.jacobi = None

        # add children list backwards
        for parent in self.parents:
            parent.children.append(self)
        
        # add this node to compute graph
        self.graph.add_node(self)

    def get_parents(self):
        """
        return all parents node of this node
        """
        return self.parents
    
    def get_children(self):
        """
        return all children node of this node
        """
        return self.children
    
    def gen_node_name(self, **kargs):
        """
        generate node name
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
        forward computation
        """
        # recursively call forward()
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()

    @abc.abstractmethod
    def compute(self):
        """
        abstract method, need to be implemented later
        compute the value of this node based on its parents
        """

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        abstract method, need to be implemented later
        compute Jacobi=d(self)/d(parent)
        """
    
    def backward(self, result):
        """
        backward computation
        """
        # Jacobi=d(result)/d(self)
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
        clear the jacobi of this node
        """
        self.jacobi = None

    def dimension(self):
        """
        return the size of this node
        """
        # we only consider 2-D matrix
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
        return the shape of this node in tuple
        """
        return self.value.shape

    def reset_value(self, recursive=True):
        """
        reset the values of this node and all its descendants
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
        init Variable
        set the value of this variable randomly if init==True
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
        set the value of this variable to given value
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim

        self.reset_value()
        self.value = value
