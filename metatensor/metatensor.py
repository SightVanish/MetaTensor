from . import core
from . import ops
from . import optimizer
from . import layer
from . import trainer

default_graph = core.default_graph
get_node_from_graph = core.get_node_from_graph
name_scope = core.name_scope
Variable = core.Variable