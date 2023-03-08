import numpy as np
from numpy import ndarray


class Optimiser(object):
    def __init__(self) -> None:
        pass
    def optimise(self,learning_rate : float,operationsData: ndarray) -> ndarray:
        for operationData in operationsData:
            for x,y in np.nditer([operationData['param'],operationData['param_grad']], op_flags =['readwrite']):
                x -= learning_rate*y       
        return operationsData

