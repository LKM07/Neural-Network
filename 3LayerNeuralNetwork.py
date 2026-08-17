import numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def derivative_output(output):
    return output*(1-output)

x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0,1,1,0]]).T

np.random.seed(0)

synapse_0=2*np.random.random((3,4))-1
synapse_1=2*np.random.random((4,1))-1

for iter in range(60000):
    l0=x
    l1=sigmoid(np.dot(l0,synapse_0))
    l2=sigmoid(np.dot(l1,synapse_1))

    l2_error=y-l2
    if(iter%10000)==0:
        print("Error:"+str(np.mean(np.abs(l2_error))))
    l2_delta=l2_error*derivative_output(l2)

    l1_error=l2_delta.dot(synapse_1.T)
    l1_delta=l1_error*derivative_output(l1)

    synapse_0 += np.dot(l0.T,l1_delta)
    synapse_1 += np.dot(l1.T,l2_delta)