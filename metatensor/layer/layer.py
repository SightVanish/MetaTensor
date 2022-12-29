from ..core import Variable
from ..ops import Add, MatMul, ReLU, Logistic, Convolve, ScalarMultiply, MaxPooling
import numpy as np

def FC(input, input_size, output_size, activation):
    """
    Full-connected layer.
    """
    weights = Variable((output_size, input_size), init=True, trainable=True)
    bias = Variable((output_size, 1), init=True, trainable=True)

    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine

def Conv(feature_maps, input_shape, num_kernels, kernel_shape, activation):
    """
    Convolution layer.
    # Parameters:
    feature_maps: list, containing multiple input feature maps
    input_shape: tuple, feature map weight and hight
    num_kernels: int, number of kernels
    kernel_shape: tuple, kernel weight and height
    activation: activation function
    """
    ones = Variable(dim=input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))
    outputs = []
    for i in range(num_kernels):
        channels = []
        for fm in feature_maps:
            kernel = Variable(dim=kernel_shape, init=True, trainable=True)
            conv = Convolve(fm, kernel)
            channels.append(conv)
        channels = Add(*channels)
        bias = ScalarMultiply(Variable(dim=(1, 1), init=True, trainable=True), ones)
        affine = Add(channels, bias)
        
        if activation == 'ReLU':
            outputs.append(ReLU(affine))
        elif activation == 'Logistic':
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)
    assert len(outputs) == num_kernels
    return outputs
        
def Pooling(features_maps, kernel_shape, stride):
    """
    Pooling layer.
    # Parameters
    feature_maps: list, containing multiple input feature maps
    kernel_shape: tuple, kernel weight and height
    stride: kernel sliding stride
    """
    outputs = []
    for fm in features_maps:
        outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))
    return outputs
