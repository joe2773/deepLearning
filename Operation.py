import numpy as np
from numpy import ndarray
from Assert import Assert

class Operation(object):
    def __init__(self) -> None:
        pass

    def forward(self, input_ : ndarray):
        '''
        This function calculates the output for a given operation
        '''
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray):
        '''
        This function calculates the gradient of the output with respect to the input, for a given operation
        '''
        Assert.same_shape(self.output,output_grad)
        self.input_grad = self._input_grad(output_grad)
        Assert.same_shape(self.input_,self.input_grad)
        return self.input_grad


    def _output(self) -> ndarray:
        '''
        This function should calculate the output of a given operation, for a given input
        '''
        raise NotImplementedError()

    def _input_grad(self, _output_grad: ndarray) -> ndarray:
        '''
        This function should calculate the gradient of the output with respect to the input, for a given operation and a given output
        '''
        raise NotImplementedError()
