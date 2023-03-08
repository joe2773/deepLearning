import numpy as np
from numpy import ndarray
from ParamOperation import ParamOperation

class BiasAdd(ParamOperation):
    def __init__(self, B : ndarray) -> ndarray :
        super().__init__(B)

    def _output(self, input: ndarray, param: ndarray) -> ndarray :
        return self.input_ + self.param

    def _input_grad(self, output_grad : ndarray) -> ndarray :
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad : ndarray) -> ndarray :
        return np.ones_like(self.param) * output_grad  