import numpy as np
from numpy import ndarray
from Operation import Operation
class Loss():
    def __init__(self) -> None:
        pass
    def _forward(self, prediction : ndarray, target : ndarray) -> float:
        self.prediction = prediction
        self.target = target
        return self._output()

    def _backward(self) -> ndarray:
        return self._input_grad()
        
    def _input_grad(self) -> ndarray:   
        raise NotImplementedError() 

    def _output(self) -> float:
        raise NotImplementedError()
