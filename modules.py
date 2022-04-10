import numpy as np
from numpy.random import rand
import Operations

class Linear:
    def __init__(self, in_dim, out_dim, autodiff):
        self.W = rand(in_dim, out_dim)
        self.b = rand(1, out_dim)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.autodiff = autodiff
    
    def forward(self, x):
        # print(x.shape)
        matmul = np.dot(x, self.W)
        self.autodiff.add([x, self.W], [None, self.dW], matmul, Operations.matmul)
        out = matmul + self.b
        self.autodiff.add([matmul, self.b], [None, self.db], out, Operations.add)
        return out

class Sigmoid:
    def __init__(self, autodiff):
        self.autodiff = autodiff
    
    def forward(self, x):
        self.x = x
        # 1 / (1+e**(-x))
        minus_1 = -1 * np.ones(x.shape)
        minus_x = minus_1 * x
        self.autodiff.add([minus_1, x], [None, None], minus_x, Operations.mul)

        ex = np.exp(minus_x)
        self.autodiff.add([minus_x], [None], ex, Operations.exp)

        denom_1 = np.ones(x.shape)
        denom = denom_1 + ex
        self.autodiff.add([denom_1, ex], [None, None], denom, Operations.add)

        num = np.ones(x.shape)
        out = num / denom
        self.autodiff.add([num, denom], [None, None], out, Operations.div)

        return out

class MSE:
    def forward(self, yhat, y, reg=False, layers=None, lambda_=0.01):
        self.yhat = yhat
        self.y = y
        self.weightLoss = 0
        out = 0
        if reg:
            self.layers = [layer for layer in layers if isinstance(layer, Linear)]
            for layer in self.layers:
                out += 1/2 * ((np.linalg.norm(layer.W))**2+(np.linalg.norm(layer.b))**2) * lambda_
                self.weightLoss += (np.linalg.norm(layer.W)+np.linalg.norm(layer.b)) * lambda_
        out = (yhat-y)**2
        return out
    
    def backward(self):
        return 2*(self.yhat - self.y) + self.weightLoss