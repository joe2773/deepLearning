import numpy as np
from numpy import ndarray
from Operation import Operation

class SigmoidOperation(Operation):
    def _output(self) -> ndarray:
        return 1/(1+np.exp(-1*self.input_))

    def _input_grad(self, _output_grad: ndarray) -> ndarray:
        sigmoid_derivative = self.output*(1-self.output)
        return sigmoid_derivative * _output_grad
