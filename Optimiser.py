import numpy as np
from numpy import ndarray


class Optimiser(object):
    def __init__(self) -> None:
        pass
    def optimise(self,learning_rate : float,operationsData: ndarray) -> ndarray:
        for operationData in operationsData:
            for (x,y) in zip(operationData['param'], operationData['param_grad']):
                print(x)
                for (i,j) in zip(x,y):
                    print(i)
                    i = i - (learning_rate * j)
                    print(i)
                print(x)
                print('---')
       
        return operationsData

