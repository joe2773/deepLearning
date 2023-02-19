import numpy as np
from numpy import ndarray


class Optimiser(object):
    def __init__(self) -> None:
        pass
    def optimise(self,learning_rate : float, params: ndarray, param_grads : ndarray) -> ndarray:
        for i, paramMatrix in params:
            for j, param in paramMatrix:
                param -= learning_rate * param_grads[i][j]
        return params

