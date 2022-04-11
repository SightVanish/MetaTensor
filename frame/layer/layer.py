from ..core import Variable
from ..ops import Add, MatMul, ReLU, Logistic

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
        print(activation + " is not supported.")
        raise ValueError
