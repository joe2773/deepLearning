import numpy as np
from numpy import ndarray
import Loss as Loss
import Optimiser as Optimiser
import Operation as Operation
import NeuralNetwork as NeuralNetwork

class Trainer(object):
    def __init__(self) -> None:
        pass
    def train(self,
        num_epochs : int,
        learning_rate : float,
        operations: ndarray([Operation]),
        inputs : ndarray,
        targets: ndarray,
        loss : Loss,
        optimiser : Optimiser):
        for i in range(num_epochs):
            network = NeuralNetwork(
                inputs,
                targets,
                operations,
                loss)
            output = network._forward()
            network._backward()
            param_grads = network.param_grads
            params = optimiser.optimise(learning_rate, params ,param_grads)
            print("For epoch: #"+ i + "Outputs are : " + output)
            print("Loss is" + network.loss)
