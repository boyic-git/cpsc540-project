import numpy as np
import Operations
import random
from AutoDiff import AutoDiff
from modules import Linear, Sigmoid, MSE
import torch
import torch.nn as nn

class NN:
    def __init__(self, in_dim, out_dim, hiddens, lr=0.01):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        self.lr = lr
        self.autoDiff = AutoDiff()
        self.loss = MSE()

        self.layers.append(Linear(in_dim, hiddens[0], self.autoDiff))
        self.layers.append(Sigmoid(self.autoDiff))
        for i in range(len(hiddens)-1):
            self.layers.append(Linear(hiddens[i], hiddens[i+1], self.autoDiff))
            self.layers.append(Sigmoid(self.autoDiff))
        self.layers.append(Linear(hiddens[-1], out_dim, self.autoDiff))
        # self.layers.append(SoftmaxCE(self.autoDiff))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def backward(self, yhat, y):
        # self.loss.forward(yhat, y, reg=True, layers=self.layers)
        self.loss.forward(yhat, y)
        loss = self.loss.backward()
        self.autoDiff.backward(loss)
    
    def step(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.dW.fill(0.0)
                layer.db.fill(0.0)
        self.autoDiff.clear()


if __name__ == "__main__":
    data = np.load("mnist35.npz")
    X = data["X"]
    y = data["y"].T
    Xtest = data["Xtest"]
    ytest = data["ytest"].T
    model = NN(784, 1, [10], 0.01)

    torchModel = nn.Sequential(nn.Linear(784, 10), nn.Sigmoid(), nn.Linear(10,1))
    torchLoss = nn.MSELoss()
    optimizer = torch.optim.SGD(torchModel.parameters(), lr=0.001, momentum=0)
    torchModel[0].weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].W.T))
    torchModel[0].bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].b))
    torchModel[2].weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].W.T))
    torchModel[2].bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].b))

    for i in range(2):
        x = random.randint(0, X.shape[0]-1)
        model.zero_grad()
        yhat = model.forward(np.array(X[[x],:]))

        xTensor = torch.DoubleTensor(np.array(X[[x],:]))
        torchyhat = torchModel(xTensor)
        output = torchLoss(torchyhat, torch.DoubleTensor([[y[x]]]))

        print(model.loss.forward(yhat, y[x]))
        model.backward(yhat, y[x])
        model.step()

        print(output)
        output.backward()
        optimizer.step()
        # if i % 500 == 0:
        #     print(i, yhat, y[x], "loss:", output)
        print("Linear1 W:", np.sum(np.abs(model.layers[0].W - torchModel[0].weight.T.detach().numpy()))) 
        print("Linear1 b:", np.sum(np.abs(model.layers[0].b - torchModel[0].bias.detach().numpy()))) 
        print("Linear2 W:", np.sum(np.abs(model.layers[2].W - torchModel[2].weight.T.detach().numpy()))) 
        print("Linear2 b:", np.sum(np.abs(model.layers[2].b - torchModel[2].bias.detach().numpy()))) 
    
    # testCorrect = 0
    # for i in range(ytest.size):
    #     yhat = -1 if model.forward(np.array(Xtest[[i],:]))[0] < 0 else 1
    #     if yhat == ytest[i]:
    #         testCorrect += 1
    # print(testCorrect/ytest.size)

    