import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mse_loss(t, f):
    return ((t-f)**2).mean()

def deriv_sigmoid(x):
    fx=sigmoid(x)
    return (fx*(1-fx))

class NeuralNetwork:
    def __init__(self):
        self.w1=np.random.normal()
        self.w2=np.random.normal()
        self.w3=np.random.normal()
        self.w4=np.random.normal()
        self.w5=np.random.normal()
        self.w6=np.random.normal()

        self.b1=np.random.normal()
        self.b2=np.random.normal()
        self.b3=np.random.normal()

    def feedforward(self, x):
        h1=sigmoid(self.w1*x[0]+self.w2*x[1]+self.b1)
        h2=sigmoid(self.w3*x[0]+self.w4*x[1]+self.b2)
        o1=sigmoid(self.w5*h1+self.w6*h2+self.b3)
        return o1

    def train(self, data, t_true):
        learn_rate=0.1

        for times in range(1000):
            for x, t in zip(data, t_true):
                sum_h1=self.w1*x[0]+self.w2*x[1]+self.b1
                h1=sigmoid(sum_h1)
                sum_h2=self.w3*x[0]+self.w4*x[1]+self.b2
                h2=sigmoid(sum_h2)
                sum_o1=self.w5*h1+self.w6*h2+self.b3
                o1=sigmoid(sum_o1)
                f=o1

                dL_df=-2*(t-f)

                df_dw5=h1*deriv_sigmoid(sum_o1)
                df_dw6=h2*deriv_sigmoid(sum_o1)
                df_db3=deriv_sigmoid(sum_o1)
                df_dh1=self.w5*deriv_sigmoid(sum_o1)
                df_dh2=self.w6*deriv_sigmoid(sum_o1)

                dh1_dw1=x[0]*deriv_sigmoid(sum_h1)
                dh1_dw2=x[1]*deriv_sigmoid(sum_h1)
                dh1_db1=deriv_sigmoid(sum_h1)

                dh2_dw3=x[0]*deriv_sigmoid(sum_h2)
                dh2_dw4=x[1]*deriv_sigmoid(sum_h2)
                dh2_db2=deriv_sigmoid(sum_h2)

                self.w1-=learn_rate*dL_df*df_dh1*dh1_dw1
                self.w2-=learn_rate*dL_df*df_dh1*dh1_dw2
                self.b1-=learn_rate*dL_df*df_dh1*dh1_db1

                self.w3-=learn_rate*dL_df*df_dh2*dh2_dw3
                self.w4-=learn_rate*dL_df*df_dh2*dh2_dw4
                self.b2-=learn_rate*dL_df*df_dh2*dh2_db2

                self.w5-=learn_rate*dL_df*df_dw5
                self.w6-=learn_rate*dL_df*df_dw6
                self.b3-=learn_rate*dL_df*df_db3

            if times%10==0:
                f_pred=np.apply_along_axis(self.feedforward, 1, data)
                loss=mse_loss(t_true, f_pred)
                print("Times %d loss: %.3f" %(times, loss))

data=np.array([
    [-6, 14],
    [4, -16],
    [-4, 9],
    [2, -11]
])

t_true=np.array([
    1,
    0,
    1,
    0
])

network=NeuralNetwork()
network.train(data, t_true)

first=np.array([-6, 14])
second=np.array([4, -16])
third=np.array([-4, 9])
fourth=np.array([2, -11])
seventh=np.array([-6, -1])
print("1st Day: %.3f" %network.feedforward(first))
print("2nd Day: %.3f" %network.feedforward(second))
print("3rd Day: %.3f" %network.feedforward(third))
print("4th Day: %.3f" %network.feedforward(fourth))
print("7th Day: %.3f" %network.feedforward(seventh))