import numpy as np
from numpy import ndarray
from Operation import Operation
from ParamOperation import ParamOperation
from MeanSquaredErrorLoss import MeanSquaredError
from SigmoidOperation import Sigmoid
from WeightMultiplyOperation import WeightMultiply
from NeuralNetwork import NeuralNetwork
from BiasAddOperation import BiasAdd




weights = np.random.rand(4,4)
inputs = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
targets = np.array([1,1,1,1])
weightMultiplyOp = WeightMultiply(weights)
loss = MeanSquaredError()
network = NeuralNetwork(
    inputs,
    targets,
    np.array([weightMultiplyOp]),
    loss)
output = network._forward()
grads = network._backward()
param_grads = network.param_grads
print(param_grads)