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
    def __init__(self,input: ndarray, target: ndarray, operationsData: ndarray, loss: Loss) -> None:
        self.operationsData = operationsData
        self.lossOperation = loss
        self.input = input
        self.target = target
        self.param_grads = np.empty([1,1])
        super().__init__()

    def _forward(self) -> ndarray:
        #This function calculates the output of the model by calculating 
        #the output of all of its constituent operations
        for operationData in self.operationsData:
            if(self._isParamOperation(operationData['operation'] == False)):
                self.input = operationData['operation']._forward(self.input)
                self.output = self.input
            else:
                self.input = operationData['operation']._forward(self.input, operationData['param'])
                self.output = self.input
            
        self.loss = self.lossOperation._forward(self.output,self.target)
        return self.output

    def _backward(self) -> ndarray:
        #This function calculates the gradient of the model by first calculating the loss_gradient
        #and then working back through the model and calculating the input gradient, alongside the param gradient for each of its operations
        #multiplying them at each step
        loss_grad = self.lossOperation._backward()

        for operationData in self.operationsData:

            if(operationData != self.operationsData[0]):
                if(self._isParamOperation(operationData['operation']) == False):
                    self.input_grad = operationData['operation']._backward(self.input_grad,operationData['param'])
                else:
                    self.input_grad = operationData['operation']._backward(self.input_grad)
                    operationData['param_grad'] = operationData['operation']._param_grad(self.input_grad)
                
            if(operationData == self.operationsData[0]):
                if(self._isParamOperation(operationData['operation']) == False):
                    self.input_grad == operationData['operation']._backward(loss_grad,operationData['param'])
                else:
                    self.input_grad = operationData['operation']._backward(loss_grad,operationData['param'])
                    operationData['param_grad'] = operationData['operation']._param_grad(loss_grad)
           
        return self.operationsData       
             
    def _isParamOperation(self, operation: Operation) -> bool:
        #This function simply checks if a given operation is a parameter operation or not
        if issubclass(operation.__class__, ParamOperation):
            return True
        return False