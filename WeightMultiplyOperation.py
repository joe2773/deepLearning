import numpy as np
from numpy import ndarray
from Assert import Assert
from ParamOperation import ParamOperation

'''
This operation is simply multiplying two matrices , typically an input X and a weight matrix W
'''
class WeightMultiply(ParamOperation):
    def __init__(self) -> None:
        pass

    def _output(self, param : ndarray) -> ndarray:
        return np.dot(self.input_, param)

    def _input_grad(self, output_grad: ndarray,param: ndarray) -> ndarray:
        return np.dot(output_grad, np.transpose(param))

    def _param_grad(self, param: ndarray, output_grad : ndarray) -> ndarray:
        return np.dot(np.transpose(self.input_),output_grad)