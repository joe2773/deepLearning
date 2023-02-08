import numpy as np
from numpy import ndarray
from Operation import Operation
from ParamOperation import ParamOperation
from Loss import Loss
from MeanSquaredErrorLoss import MeanSquaredError
from SigmoidOperation import Sigmoid
from WeightMultiplyOperation import WeightMultiply
from BiasAddOperation import BiasAdd

class NeuralNetwork(object):
    def __init__(self,input: ndarray, target: ndarray, operations: ndarray, loss: Loss) -> None:
        self.operations = operations
        self.lossOperation = loss
        self.input = input
        self.target = target
        super().__init__()

    def _forward(self) -> ndarray:
        #This function calculates the output of the model by calculating 
        #the output of all of its constituent operations
        for operation in self.operations:
            self.input = operation.forward(self.input)
            self.output = self.input

        #self.loss = self.lossOperation._forward(self.output,self.target)
        return self.output

    def _backward(self) -> ndarray:
        #This function calculates the gradient of the model by first calculating the loss_gradient
        #and then working back through the model and calculating the input gradient, alongside the param gradient for each of its operations
        #multiplying them at each step
        loss_grad = self.loss._input_grad()
        for operation in self.operations:
            if(self.operations[0] == operation):
                self.input_grad = operation._input_grad(loss_grad)
            if(self._isParamOperation()):
                self.input_grad = operation._input_grad(self.input_grad)
                self.param_grads.append(operation._param_grad(self.input_grad))
                continue
            self.input_grad = operation._input_grad(self.input_grad)
        return self.input_grad        
             
    def _isParamOperation(self, operation: Operation) -> bool:
        #This function simply checks if a given operation is a parameter operation or not
        if issubclass(operation.__class__, ParamOperation):
            return True
        return False