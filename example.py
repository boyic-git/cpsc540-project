from time import sleep
import numpy as np

import Operations
import random
from AutoDiff import AutoDiff
from modules import Linear, Sigmoid, MSE
from NN import NN
import matplotlib.pyplot as plt

data = np.load("basisData.npz")
X = data["X"]
y = data["y"]
xVals = np.array([np.linspace(-10, 10, retstep=0.05)[0]]).T

model = NN(1, 1, [10,5], lr=0.1)
# yhat = model.forward(xVals)
# print(yhat)
# plt.figure()
# plt.plot(xVals, yhat, "r-")
# plt.plot(X, y, ".")
# # plt.draw()
# plt.pause(0.01)
# plt.show()
# plt.clf()

for i in range(100000):
    k = np.random.randint(0, X.shape[0])
    model.zero_grad()

    yhat = model.forward(np.array([X[k,:]]))

    model.backward(yhat, [y[k]])
    model.step()
    # break

    
    # break
    if i % 500 == 0:
        # print(i, yhat, y[k], model.loss.forward(yhat, [y[k]])[0][0])

        yhat = model.forward(xVals)
        plt.plot(xVals, yhat, "r-")
        plt.plot(X, y, ".")
        plt.draw()
        plt.pause(0.01)
        plt.clf()
plt.show()