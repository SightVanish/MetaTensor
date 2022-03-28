from frame.core.node import Variable, name_scope
from frame.core.utils import get_node_from_graph
from . import core
from . import ops

default_graph = core.default_graph
get_node_from_graph = core.get_node_from_graph
name_scope = core.name_scope
Variable = core.Variable