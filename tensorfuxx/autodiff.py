#from __future__ import division
import numpy as np
from tensorfuxx.Clink import *
init_value = {}
class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.
            
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's
            # const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, rhs):
        if isinstance(rhs, Node):
            new_node = sub_op(self, rhs)
        else:
            new_node = add_op(self, neg_op(constant(rhs)))
        return new_node

    def __rsub__(self, lhs):
        if isinstance(lhs, Node):
            new_node = sub_op(lhs, self)
        else:
            new_node = add_byconst_op(neg_op(self),lhs)
        return new_node

    def __div__(self, rhs):
        if isinstance(rhs, Node):
            new_node = div_op(self, rhs)
        else:
            new_node = div_op(self, constant(rhs))
        return new_node

    def __rdiv__(self, lhs):
        if isinstance(lhs, Node):
            new_node = div_op(lhs, self)
        else:
            new_node = div_op(constant(lhs), self)
        return new_node

    def __str__(self):
        """Allow print to display node name.""" 
        return self.name

    __repr__ = __str__

    __floordiv__ = __div__
    __rfloordiv__ = __rdiv__

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __neg__(self):
        return neg_op(self)

    def eval(self, feed_dict={}):
        from tensorfuxx.methods import Executor
        ex = Executor(eval_node_list=[self])
        return ex.run(feed_dict=feed_dict)[0]

    run = eval

class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.
        
        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

class ExtendToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        ans = input_vals[0]
        shape = input_vals[1].shape
        try:
            ans = np.broadcast_to(ans, shape)
        except:            
            shape2 = ans.shape
            while len(shape2) < len(shape):
                shape2 = shape2 + (1,)
            ans.resize(shape2)
            ans = np.broadcast_to(ans, shape)
            return ans
        else:
            return ans

    def gradient(self, node, output_grad):
        return [reduce_shape_op(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]

class NegOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return 0. - input_vals[0]

    def gradient(self, node, output_grad):
        return [neg_op(output_grad)]

class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [reduce_shape_op(output_grad, node.inputs[0]), reduce_shape_op(output_grad, node.inputs[1])]

class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [reduce_shape_op(output_grad, node.inputs[0])]

class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        return [reduce_shape_op(output_grad, node.inputs[0]), reduce_shape_op(neg_op(output_grad), node.inputs[1])]

class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        new_node1 = mul_op(output_grad,node.inputs[1])
        new_node2 = mul_op(output_grad,node.inputs[0])
        return [reduce_shape_op(new_node1, node.inputs[0]), reduce_shape_op(new_node2, node.inputs[1])]

class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        new_node = mul_byconst_op(output_grad,node.const_attr)
        return [reduce_shape_op(new_node, node.inputs[0])]

class DivOp(Op):  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [reduce_shape_op(output_grad / node.inputs[1], node.inputs[0]),reduce_shape_op((neg_op(output_grad * node.inputs[0]) / node.inputs[1]) / node.inputs[1], node.inputs[1])]

class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp(node.inputs[0])]

class LogOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[0]]

class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        A = input_vals[0]
        B = input_vals[1]
        n=A.shape[0]
        m=B.shape[1]
        if(node.matmul_attr_trans_A): n=A.shape[1]
        if(node.matmul_attr_trans_B): m=B.shape[0]
        C = np.zeros([n,m], dtype = np.float32)
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        c_kernel.mat_mul(get_pointer(A),A.shape[0],A.shape[1],
            get_pointer(B),B.shape[0],B.shape[1],
            get_pointer(C),n,m,int(node.matmul_attr_trans_A),int(node.matmul_attr_trans_B))

        return C

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
            
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        new_node1 = matmul(output_grad,node.inputs[1],False,True)
        new_node2 = matmul(node.inputs[0],output_grad,True,False)
        return [new_node1, new_node2]

class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self, dtype=None, shape=None, name="Placeholder"):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        #assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class VariableOp(Op):
    def __call__(self, value, dtype=None, shape=None, name="Variable"):
        new_node = Op.__call__(self)
        if isinstance(value,Node):
            init_value[new_node] = value.const_attr
        elif not isinstance(value,np.ndarray):
            init_value[new_node] = np.array(value)
        else:
            init_value[new_node] = value
        return new_node

    def compute(self, node, input_vals):
        return node.const_attr

    def gradient(self, node, output_grad):
        return None

class GlobalVariablesInitializerOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        for node, value in init_value.items():
            node.const_attr = value
        return True

class ConstantOp(Op):
    def __call__(self, initial_value, dtype=None, shape=None, name="Const"):
        new_node = Op.__call__(self)
        if not isinstance(initial_value, np.ndarray) and (shape is not None):
            initial_value = np.ones(shape=shape) * initial_value
        new_node.const_attr = np.array(initial_value).reshape(shape).astype(dtype)
        return new_node

    def compute(self, node, input_vals):
        return node.const_attr

    def gradient(self, node, output_grad):
        return None

class AssignOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        if not isinstance(node_B, Node):
            node_B = constant(node_B)
        new_node.inputs = [node_B]
        new_node.const_attr = node_A
        return new_node

    def compute(self, node, input_vals):
        node.const_attr.const_attr = input_vals[0]
        return input_vals[0]

class ReduceSumOp(Op):
    def __call__(self, node_A, axis=None, keep_dims=False, reduction_indices=None):
        new_node = Op.__call__(self)
        if axis is None and reduction_indices is not None:
            axis = tuple(reduction_indices)
        new_node.inputs = [node_A]
        new_node.const_attr = (axis, keep_dims)
        return new_node

    def compute(self, node, input_vals):
        return np.sum(input_vals[0], axis=node.const_attr[0], keepdims=node.const_attr[1])

    def gradient(self, node, output_grad):
        return [extend_to(output_grad,node.inputs[0])]

class ReduceMeanOp(Op):
    def __call__(self, node_A, axis=None, keep_dims=False, reduction_indices=None):
        new_node = Op.__call__(self)
        if axis is None and reduction_indices is not None:
            axis = tuple(reduction_indices)
        new_node.inputs = [node_A]
        new_node.const_attr = (axis, keep_dims)
        return new_node

    def compute(self, node, input_vals):
        return np.mean(input_vals[0], axis=node.const_attr[0], keepdims=node.const_attr[1])

    def gradient(self, node, output_grad):
        return [reduce_shape_op(extend_to(output_grad,node.inputs[0])/reduce_sum(oneslike_op(node.inputs[0]), axis=node.const_attr[0], keep_dims=node.const_attr[1]), node.inputs[0])]

class EqualOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return np.equal(input_vals[0], input_vals[1])

class ArgMaxOp(Op):
    def __call__(self, node_A, axis=None, name=None, dimension=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = axis
        return new_node

    def compute(self, node, input_vals):
        return np.argmax(input_vals[0], axis=node.const_attr)

class CastOp(Op):
    def __call__(self, node_A, dtype, name=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = dtype
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0].astype(node.const_attr)

class PackOp(Op):
    def __call__(self, node_list, name=None):
        new_node = Op.__call__(self)
        new_node.inputs = node_list
        if(name): new_node.name=name
        return new_node

    def compute(self, node, input_vals):
        return None

class ReduceShapeOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        ans = input_vals[0]
        shape = input_vals[1].shape
        while len(ans.shape)>len(shape):
            ans = np.sum(ans, axis=0)
        
        for dim in range(len(shape)):
            if ans.shape[dim] > input_vals[1].shape[dim]:
                ans = np.sum(ans, axis=dim, keepdims=True)

        return ans

    def gradient(self, node, output_grad):
        return [extend_to(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]

class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return (np.abs(input_vals[0]) + input_vals[0]) / 2

    def gradient(self, node, output_grad):
        return [relu_grad_op(node.inputs[0], output_grad)]

class ReluGradOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]

class SqrtOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals):
        return np.sqrt(input_vals[0])

    def gradient(self, node, output_grad):
        raise NotImplementedError

class PowOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.pow(node_A, node_B)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        output_val = np.power(input_vals[0], input_vals[1])
        return output_val

    def gradient(self, node, output_grad):
        raise NotImplementedError

class ReshapeOp(Op):
    def __call__(self, node_A, shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = shape
        return new_node

    def compute(self, node, input_vals):
        return np.reshape(input_vals[0], tuple(node.const_attr))

    def gradient(self, node, output_grad):
        return [reshape_to(output_grad, node.inputs[0])]

class ReshapeToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return np.reshape(input_vals[0], input_vals[1].shape)

    def gradient(self, node, output_grad):
        return [reshape_to(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]

class RandomZoOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        return (np.random.uniform(size=input_vals[0].shape) < input_vals[1])

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0]), zeroslike_op(node.inputs[1])]

# Create global singletons of operators.
neg_op = NegOp()
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
div_op = DivOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul = MatMulOp()
exp = ExpOp()
log = LogOp()
placeholder_op = PlaceholderOp()
placeholder = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
Variable = VariableOp()
global_variables_initializer = GlobalVariablesInitializerOp()
constant = ConstantOp()
assign = AssignOp()
reduce_sum = ReduceSumOp()
reduce_mean = ReduceMeanOp()
equal = EqualOp()
argmax = ArgMaxOp()
cast = CastOp()
pack = PackOp()
reduce_shape_op = ReduceShapeOp()
extend_to = ExtendToOp()
relu_grad_op = ReluGradOp()
sqrt_op = SqrtOp()
pow_op = PowOp()
reshape = ReshapeOp()
reshape_to = ReshapeToOp()
random_zo_op = RandomZoOp()