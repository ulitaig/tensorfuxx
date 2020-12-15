import numpy as np
from tensorfuxx.nn import *

class Executor:
    """Executor computes values for a given subset of nodes in a computation graph.""" 
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.
        
        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        if(Check_Fcc(self)):
            #print("fuck")
            #raise
            return [0]
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all
        # nodes.
        for node, value in node_to_val_map.items():
            if not isinstance(value,np.ndarray):
                node_to_val_map[node] = np.array(value)

        topo_order = find_topo_sort(self.eval_node_list)
        
        for node in topo_order:
            if node not in feed_dict:
                node_to_val_map[node] = node.op.compute(node,[node_to_val_map[ns] for ns in node.inputs])
                #print(node.name)

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results

def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as
    # oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node.  But this is the common case for loss
    # function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we
    # are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad
        input_grads = node.op.gradient(node,grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = [input_grads[i]]
            else:
                node_to_output_grads_list[node.inputs[i]].append(input_grads[i])

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list



def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None, name=None):
    if(seed):
        np.random.seed(seed)
    return np.random.normal(loc=mean, scale=stddev, size=shape).astype(dtype)

