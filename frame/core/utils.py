from numpy import isin
from .node import Variable
from .graph import default_graph

def get_node_from_graph(node_name, name_scope=None, graph=None):
    """
    Return node based on graph and node_name. If there is no such node, return None.
    """
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None

def get_trainable_variables_from_graph(node_name=None, name_scope=None, graph=None):
    """
    Return node based on graph and node_name. If node_name is None, return all trainable variable nodes.
    """
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in default_graph.nodes if isinstance(node, Variable) and node.trainable]
    
    node = get_node_from_graph(node_name, name_scope, graph=graph)

    if node is None:
        return None
    assert isinstance(node, Variable) and node.trainable
    return node

def update_node_value_in_graph(node_name, new_value, name_scope=None, graph=None):
    node = get_node_from_graph(node_name, name_scope, graph)

    assert isinstance(node, Variable) and node is not None
    assert node.value.shape == new_value.shape
    
    node.value = new_value
    