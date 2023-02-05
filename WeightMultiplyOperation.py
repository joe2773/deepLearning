import numpy as np
from numpy import ndarray
from Assert import Assert
from ParamOperation import ParamOperation

'''
This operation is simply multiplying two matrices , typically an input X and a weight matrix W
'''
class WeightMultiply(ParamOperation):
    def __init__(self, W: ndarray) -> None:
        super().__init__(W)

    def _output(self) -> ndarray:
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, np.transpose(self.param(1,0)))

    def _param_grad(self, output_grad : ndarray) -> ndarray:
        return np.dot(np.transpose(self.input_,(1,0)),output_grad)