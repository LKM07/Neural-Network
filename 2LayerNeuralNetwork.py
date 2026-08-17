import numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def derivative_output(output):
    return output*(1-output)

x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0,0,1,1]]).T

np.random.seed(0)

synapse_0=2*np.random.random((3,1))-1

for iter in range(10000):
    l0=x
    l1=sigmoid(np.dot(l0,synapse_0))

    l1_error=y-l1
    l1_delta=l1_error*derivative_output(l1)

    synapse_0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(l1)