import numpy as np
from numpy import ndarray
from Operation import Operation
from ParamOperation import ParamOperation
from MeanSquaredErrorLoss import MeanSquaredError
from SigmoidOperation import Sigmoid
from WeightMultiplyOperation import WeightMultiply
from NeuralNetwork import NeuralNetwork
from BiasAddOperation import BiasAdd
from Optimiser import Optimiser


optimiser = Optimiser()
weights = np.random.rand(4,4)
inputs = np.array([[3,4,5,6],[1,2,3,4],[1,6,4,4],[3,7,3,4]])
targets = np.random.rand(4,4)
num_epochs = 1
learning_rate = 0.1

weightMultiply = WeightMultiply()
operationsData = np.array([
    {'operation': weightMultiply, 'param' : np.random.rand(4,4), 'param_grad' : np.zeros((4,4), dtype=float)},
    ])

for i in range(num_epochs):
    
    lossOperation = MeanSquaredError()
    network = NeuralNetwork(
        inputs,
        targets,
        operationsData,
        lossOperation)
    output = network._forward()
    operationsData = network._backward()
    operationsData = optimiser.optimise(learning_rate,operationsData)

#print('output:')
print(output)
#print('inputs:')
#print(inputs)
#print('targets:')
#print(targets)
#print(operationsData[0])
