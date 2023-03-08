import numpy as np
from numpy import ndarray
from Operation import Operation
from Assert import Assert
class ParamOperation(Operation):
    def __init__(self) -> ndarray:
        super().__init__()
        pass

    def _forward(self, input : ndarray, param: ndarray):
        self.input = input
        self.output = self._output(input,param)
        return self.output
    
    def _backward(self, output_grad :ndarray, param: ndarray) -> ndarray:    
        self.input_grad = self._input_grad(output_grad, param)
        return self.input_grad

    def _param_grad(self, output_grad: ndarray)-> ndarray:
        '''
        This function should calculate the gradient of the output with respect to the param, for a given operation and input
        '''
        raise NotImplementedError();