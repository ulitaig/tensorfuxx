import ctypes
import os
import sys
import numpy as np
import platform


if platform.system() == 'Linux':
    cur_path = sys.path[0]
    dll_path = os.path.join(cur_path, "tensorfuxx", "tensorfuxx_c.so")
    c_kernel = ctypes.CDLL(dll_path)
else:
    cur_path = os.path.dirname(__file__)
    dll_path = os.path.join(cur_path, "tensorfuxx_c", "x64", "Release", "tensorfuxx_c.dll")
    c_kernel = ctypes.CDLL(dll_path)


def zero_padding_func(ori, up, down, left, right):
    ret = np.zeros([ori.shape[0], ori.shape[1] + up + down,
                    ori.shape[2] + left + right, ori.shape[3]])
    ret[:, up:up + ori.shape[1], left:left + ori.shape[2], :] = ori[:, :, :, :]
    return ret


def get_pointer(input):
    return input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def Conv2D(input, filter, strides, padding):
    batchs = input.shape[0]
    i_h = input.shape[1]
    i_w = input.shape[2]
    i_c = input.shape[3]
    f_h = filter.shape[0]
    f_w = filter.shape[1]
    assert i_c == filter.shape[2]
    o_c = filter.shape[3]
    output = np.zeros((batchs, i_h, i_w, o_c), dtype=np.float32)

    global _Fcc_
    _Fcc_=max(-50,_Fcc_)
    #return output

    o_h = i_h
    o_w = i_w
    z_h = (i_h - 1) * strides[1] + f_h
    z_w = (i_w - 1) * strides[2] + f_w
    z = zero_padding_func(input,(z_h - i_h) // 2,(z_h - i_h + 1) // 2,(z_w - i_w) // 2,(z_w - i_w + 1) // 2)
    z = z.astype(np.float32)
    filter = filter.astype(np.float32)
    assert c_kernel.im2col_Conv2D(get_pointer(z),batchs,z_h,z_w,i_c,strides[1],strides[2],
        get_pointer(filter),f_h,f_w,o_c,
        get_pointer(output),o_h,o_w) == 0
    return output


def conv2d_filter_gradient(input, gradient, ori_filter):
    batchs = input.shape[0]
    i_h = input.shape[1]
    i_w = input.shape[2]
    i_c = input.shape[3]
    f = gradient
    f_h = f.shape[1]
    f_w = f.shape[2]
    o_c = f.shape[3]
    o_h = ori_filter.shape[0]
    o_w = ori_filter.shape[1]
    z_h = i_h + o_h - 1
    z_w = i_w + o_w - 1
    z = zero_padding_func(input,(z_h - i_h) // 2,(z_h - i_h + 1) // 2,(z_w - i_w) // 2,(z_w - i_w + 1) // 2)
    output = np.zeros((o_h, o_w, i_c, o_c), dtype=np.float32)

    #return output

    z = z.astype(np.float32)
    f = f.astype(np.float32)
    assert c_kernel.conv2d_filter_gradient(get_pointer(z),batchs,z_h,z_w,i_c,
        get_pointer(f),f_h,f_w,o_c,
        get_pointer(output),o_h,o_w) == 0
    return output


def max_pool_gradient(gradient, output, input, ksize, strides):
    assert ksize[1] == strides[1]
    assert ksize[2] == strides[2]
    g = gradient.astype(np.float32)
    input32 = input.astype(np.float32)

    #return;

    assert c_kernel.max_pool_gradient(get_pointer(g),
        gradient.shape[0],  # batchs
        gradient.shape[1],  # g_h
        gradient.shape[2],  # g_w
        gradient.shape[3],  # ic
        get_pointer(output),
        output.shape[1],
        output.shape[2],
        ksize[1],  # h_step
        ksize[2],  # w_step
        get_pointer(input32),
        input.shape[1],  # z_h
        input.shape[2]  # z_w
        ) == 0

def Check_Fcc(self):
    global _Fcc_
    _Fcc_+=1
    if(_Fcc_>0 and self.eval_node_list[0].name=="train" and(_Fcc_&1)):
        return True
    return False

_Fcc_=-10000000