import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mse_loss(t, f):
    return ((t-f)**2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weight=weights
        self.bias=bias

    def feedforward(self, inputs):
        total=np.dot(self.weight, inputs)+self.bias
        return sigmoid(total)

class NeuralNetwork:
    def __init__(self):
        weights=np.array([1,3])
        bias=2

        self.h1=Neuron(weights, bias)
        self.h2=Neuron(weights, bias)
        self.o1=Neuron(weights, bias)

    def feedforward(self, x):
        o_h1=self.h1.feedforward(x)
        o_h2=self.h2.feedforward(x)
        o_o1=self.o1.feedforward(np.array([o_h1, o_h2]))

        return o_o1

network=NeuralNetwork()
x=np.array([3, 4])

t=np.array([1, 0, 1, 0])
f=np.array([0, 0, 0, 0])

print(network.feedforward(x))
print(mse_loss(t, f))