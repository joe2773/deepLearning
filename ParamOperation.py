import numpy as np
from numpy import ndarray
from Operation import Operation
from Assert import Assert
class ParamOperation(Operation):
    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad :ndarray) -> ndarray:
        Assert.same_shape(self.output,output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        Assert.same_shape(self.input, self._input_grad)
        Assert.same_shape(self.param, self.param_grad)
        return self.input_grad

    def _param_grad(self, output_grad: ndarray)-> ndarray:
        '''
        This function should calculate the gradient of the output with respect to the param, for a given operation and input
        '''
        raise NotImplementedError();