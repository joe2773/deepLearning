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
inputs = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
targets = np.array([1,1,1,1])
params = np.array(np.random.rand(4,4),np.ones_like(1,4))
num_epochs = 100
learning_rate = 0.01
for i in range(num_epochs):
    weightMultiplyOp = WeightMultiply(params[0])
    biasAddOp = BiasAdd(params[1])
    loss = MeanSquaredError()
    network = NeuralNetwork(
        inputs,
        targets,
        np.array(weightMultiplyOp,biasAddOp),
        loss)
    output = network._forward()
    grads = network._backward()
    param_grads = network.param_grads
    params = optimiser.optimise(learning_rate, params ,param_grads)