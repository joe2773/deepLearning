import numpy as np
from numpy import ndarray
from Loss import Loss
class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()  

    def _output(self) -> float:
        return (self.prediction - self.target)**2 / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        return 2*(self.prediction - self.target) / self.prediction.shape[0]
