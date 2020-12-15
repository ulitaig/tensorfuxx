from tensorfuxx.autodiff import *


class SoftmaxOp(Op):
    def __call__(self, node_A, dim=-1, name=None):
        exp_node_A = exp(node_A)
        new_node = exp_node_A/reduce_sum(exp_node_A, axis=dim, keep_dims=True)
        new_node.name = name
        return new_node

class SoftmaxCrossEntropyWithLogitsOp(Op):
    def __call__(self, logits, labels):
        return (-reduce_sum(labels * log(softmax(logits)), reduction_indices=[1]))

class Conv2DOp(Op):
    def __call__(self, node_A, node_B, strides=[1, 1, 1, 1], padding='SAME'):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.const_attr = (strides, padding)
        new_node.name = "Conv2D(%s,filter=%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        return Conv2D(input_vals[0],input_vals[1],node.const_attr[0],node.const_attr[1])

    def gradient(self, node, output_grad):
        return [conv2d_g_A(node.inputs[0], node.inputs[1], output_grad, node.const_attr),
                conv2d_g_B(node.inputs[0], node.inputs[1], output_grad, node.const_attr)]


class Conv2DGradientNodeAOp(Op):
    def __call__(self,  node_A, node_B, node_grad, stridesAndPadding):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_grad]
        new_node.const_attr = stridesAndPadding
        return new_node

    def compute(self, node, input_vals):
        return Conv2D(input_vals[2],np.rot90(np.transpose(input_vals[1], (0, 1, 3, 2)), axes=(0, 1), k=2),[1, 1, 1, 1],node.const_attr[1])



class Conv2DGradientNodeBOp(Op):
    def __call__(self, node_A, node_B, node_grad, stridesAndPadding):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_grad]
        new_node.const_attr = stridesAndPadding
        return new_node

    def compute(self, node, input_vals):
        return conv2d_filter_gradient(input_vals[0],input_vals[2],input_vals[1])


class MaxPoolOp(Op):
    def __call__(self, node_A, ksize, strides, padding='SAME'):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "MaxPool(%s)" % (node_A.name)
        new_node.const_attr = (ksize, strides, padding)
        return new_node

    # profile
    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        batchs = input_vals[0].shape[0]
        i_h = input_vals[0].shape[1]
        i_w = input_vals[0].shape[2]
        i_c = input_vals[0].shape[3]
        strides = node.const_attr[1]
        ksize = node.const_attr[0]
        o_h = (i_h - 1) // strides[1] + 1
        o_w = (i_w - 1) // strides[2] + 1
        z_h = ((i_h - 1) // strides[1]) * strides[1] + ksize[1]
        z_w = ((i_w - 1) // strides[2]) * strides[2] + ksize[2]
        up = (z_h - i_h) // 2
        left = (z_w - i_w) // 2
        z = zero_padding_func(input_vals[0],up,(z_h - i_h + 1) // 2,left,(z_w - i_w + 1) // 2)
        z = z.astype(np.float32)
        output = np.zeros([batchs, o_h, o_w, i_c], dtype = np.float32)
        c_kernel.max_pool(get_pointer(z),batchs,z_h,z_w,i_c,
                          get_pointer(output),o_h,o_w,strides[1],strides[2],up,left)

        return output

    def gradient(self, node, output_grad):
        return [max_pool_g(node.inputs[0], output_grad, node.const_attr)]


class MaxPoolGradientOp(Op):
    def __call__(self, node_A, node_grad, const_attr):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_grad]
        new_node.const_attr = const_attr
        return new_node

    # profile
    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        batchs = input_vals[0].shape[0]
        i_h = input_vals[0].shape[1]
        i_w = input_vals[0].shape[2]
        i_c = input_vals[0].shape[3]
        strides = node.const_attr[1]
        ksize = node.const_attr[0]
        o_h = (i_h - 1) // strides[1] + 1
        o_w = (i_w - 1) // strides[2] + 1
        z_h = ((i_h - 1) // strides[1]) * strides[1] + ksize[1]
        z_w = ((i_w - 1) // strides[2]) * strides[2] + ksize[2]
        z = zero_padding_func(ori=input_vals[0], up=(z_h - i_h) // 2, down=(z_h - i_h + 1) // 2,
                                left=(z_w - i_w) // 2, right=(z_w - i_w + 1) // 2)
        output_val = np.zeros((batchs, z_h, z_w, i_c), dtype=np.float32)
        up = (z_h - i_h) // 2
        left = (z_w - i_w) // 2
        max_pool_gradient(input_vals[1],output_val,z,ksize,strides)
        output_val = output_val[:, up:up + i_h, left:left + i_w, :]
        return output_val

class DropoutOp(Op):
    def __call__(self, node_A, node_B):
        new_node = mul_op(node_A, random_zo_op(node_A, node_B)) / node_B
        name = "Dropout(%s,prob=%s)" % (node_A.name, node_B.name)
        return new_node

def get_patch(ori, i, j, f_h, f_w, strides, i_c=None):
    if i_c is None:
        return ori[:, i * strides[1]:i * strides[1] + f_h, j * strides[2]:j * strides[2] + f_w, :]
    else:
        return ori[:, i * strides[1]:i * strides[1] + f_h, j * strides[2]:j * strides[2] + f_w, i_c]




relu = ReluOp()
softmax = SoftmaxOp()
softmax_cross_entropy_with_logits = SoftmaxCrossEntropyWithLogitsOp()
conv2d = Conv2DOp()
conv2d_g_A = Conv2DGradientNodeAOp()
conv2d_g_B = Conv2DGradientNodeBOp()
max_pool = MaxPoolOp()
max_pool_g = MaxPoolGradientOp()
dropout = DropoutOp()