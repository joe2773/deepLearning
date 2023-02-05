import numpy as np
from numpy import ndarray
from Loss import Loss
class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()  

    def _output(self) -> float:
        return (self.predictions - self.target)**2 / self.predictions.shape[0]

    def _input_grad(self) -> ndarray:
        return 2*(self.predictions - self.target) / self.predictions.shape[0]
