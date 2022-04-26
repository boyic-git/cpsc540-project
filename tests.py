import torch
import numpy as np

import Operations
import random
from AutoDiff import AutoDiff
from modules import Linear, Sigmoid, MSE
from NN import NN

def testLinear():
    x = np.random.random((1, 100))
    autoDiff = AutoDiff()

    linear = Linear(100, 1, autoDiff)
    out = linear.forward(x)

    torchLinear = torch.nn.Linear(100, 1)
    torchLinear.weight = torch.nn.Parameter(torch.DoubleTensor(linear.W.T))
    torchLinear.bias = torch.nn.Parameter(torch.DoubleTensor(linear.b))
    torchX = torch.DoubleTensor(x)
    torchOut = torchLinear(torchX)

    # print(np.sum(np.abs(out - torchOut.detach().numpy()))) 
    assert np.sum(np.abs(out - torchOut.detach().numpy())) < 1e-10, "Linear forward test failed"
    print("Linear forward test passed")

def testLinearBackward():
    x = np.random.random((1, 100))
    autoDiff = AutoDiff()

    linear = Linear(100, 1, autoDiff)
    out = linear.forward(x)
    autoDiff.backward(1)

    torchLinear = torch.nn.Linear(100, 1)
    torchLinear.weight = torch.nn.Parameter(torch.DoubleTensor(linear.W.T))
    torchLinear.bias = torch.nn.Parameter(torch.DoubleTensor(linear.b))
    torchX = torch.DoubleTensor(x)
    torchOut = torchLinear(torchX)
    torchOut.backward()

    # print("dW:", np.sum(np.abs(linear.dW - torchLinear.weight.grad.T.detach().numpy()))) 
    # print("db:", np.sum(np.abs(linear.db - torchLinear.bias.grad.detach().numpy()))) 

    assert np.sum(np.abs(linear.dW - torchLinear.weight.grad.T.detach().numpy())) < 1e-10, "Linear backward test failed"
    assert np.sum(np.abs(linear.db - torchLinear.bias.grad.detach().numpy())) < 1e-10, "Linear backward test failed"
    print("Linear backward test passed")

    # print("W:", np.sum(np.abs(linear.W - torchLinear.weight.T.detach().numpy()))) 
    # print("b:", np.sum(np.abs(linear.b - torchLinear.bias.detach().numpy()))) 

def testNN():
    x = np.random.random((1, 100))
    model = NN(100, 1, [50])

    out = model.forward(x)
    # print(out)

    torchLinear1 = torch.nn.Linear(100, 1)
    torchLinear1.weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].W.T))
    torchLinear1.bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].b))
    torchX = torch.DoubleTensor(x)
    torchOut1 = torchLinear1(torchX)

    torchSigmoid = torch.nn.Sigmoid()
    torchOut2 = torchSigmoid(torchOut1)

    torchLinear2 = torch.nn.Linear(100, 50)
    torchLinear2.weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].W.T))
    torchLinear2.bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].b))
    torchOut3 = torchLinear2(torchOut2)

    assert np.sum(np.abs(out - torchOut3.detach().numpy())) < 1e-10, "NN forward test failed"
    print("NN forward test passed")

def testNNBackward():
    x = np.random.random((1, 1))
    autoDiff = AutoDiff()
    model = NN(1, 1, [1])

    out = model.forward(x)
    model.autoDiff.backward(1)

    torchLinear1 = torch.nn.Linear(1, 1)
    torchLinear1.weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].W.T))
    torchLinear1.bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].b))
    torchX = torch.DoubleTensor(x)
    torchOut1 = torchLinear1(torchX)

    torchSigmoid = torch.nn.Sigmoid()
    torchOut2 = torchSigmoid(torchOut1)

    torchLinear2 = torch.nn.Linear(1, 1)
    torchLinear2.weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].W.T))
    torchLinear2.bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].b))
    torchOut3 = torchLinear2(torchOut2)
    torchOut3.backward()

    assert np.sum(np.abs(model.layers[0].dW - torchLinear1.weight.grad.T.detach().numpy())) < 1e-10, \
        "NN backward test failed, linear 1 dW is different"
    assert np.sum(np.abs(model.layers[0].db - torchLinear1.bias.grad.detach().numpy())) < 1e-10, \
        "NN backward test failed, linear 1 db is different"

    assert np.sum(np.abs(model.layers[2].dW - torchLinear2.weight.grad.T.detach().numpy())) < 1e-10, \
        "NN backward test failed, linear 1 dW is different"
    assert np.sum(np.abs(model.layers[2].db - torchLinear2.bias.grad.detach().numpy())) < 1e-10, \
        "NN backward test failed, linear 1 db is different"

    print("NN backward test passed")

    # print("W:", np.sum(np.abs(linear.W - torchLinear.weight.T.detach().numpy()))) 
    # print("b:", np.sum(np.abs(linear.b - torchLinear.bias.detach().numpy()))) 

def testSigmoid():
    x = np.random.random((1, 100))
    autoDiff = AutoDiff()

    linear = Linear(100, 1, autoDiff)
    sigmoid = Sigmoid(autoDiff)
    out = sigmoid.forward(linear.forward(x))

    torchLinear = torch.nn.Linear(100, 1)
    torchLinear.weight = torch.nn.Parameter(torch.DoubleTensor(linear.W.T))
    torchLinear.bias = torch.nn.Parameter(torch.DoubleTensor(linear.b))
    torchSigmoid = torch.nn.Sigmoid()
    torchX = torch.DoubleTensor(x)
    torchOut = torchSigmoid(torchLinear(torchX))

    assert np.sum(np.abs(out - torchOut.detach().numpy())) < 1e10, "Sigmoid forward test failed"
    print("Sigmoid forward test passed")

def testSigmoidBackward():
    x = np.random.random((1, 100))
    autoDiff = AutoDiff()

    linear = Linear(100, 1, autoDiff)
    sigmoid = Sigmoid(autoDiff)
    out = sigmoid.forward(linear.forward(x))
    autoDiff.backward(1)

    torchLinear = torch.nn.Linear(100, 1)
    torchLinear.weight = torch.nn.Parameter(torch.DoubleTensor(linear.W.T))
    torchLinear.bias = torch.nn.Parameter(torch.DoubleTensor(linear.b))
    torchSigmoid = torch.nn.Sigmoid()
    torchX = torch.DoubleTensor(x)
    torchOut = torchSigmoid(torchLinear(torchX))
    torchOut.backward()

    assert np.sum(np.abs(linear.dW - torchLinear.weight.grad.T.detach().numpy())) < 1e-10, \
        "Sigmoid backward test failed, dW is different"
    assert np.sum(np.abs(linear.db - torchLinear.bias.grad.detach().numpy())) < 1e-10, \
        "Sigmoid backward test failed, db is different"
    print("Sigmoid backward test passed")

def mnist35Test():
    data = np.load("mnist35.npz")
    X = data["X"]
    y = data["y"].T
    Xtest = data["Xtest"]
    ytest = data["ytest"].T
    model = NN(784, 1, [10], 0.01)

    torchModel = torch.nn.Sequential(torch.nn.Linear(784, 10), torch.nn.Sigmoid(), torch.nn.Linear(10,1))
    torchLoss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(torchModel.parameters(), lr=0.001, momentum=0)
    torchModel[0].weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].W.T))
    torchModel[0].bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[0].b))
    torchModel[2].weight = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].W.T))
    torchModel[2].bias = torch.nn.Parameter(torch.DoubleTensor(model.layers[2].b))

    for i in range(10):
        x = random.randint(0, X.shape[0]-1)
        model.zero_grad()
        yhat = model.forward(np.array(X[[x],:]))

        xTensor = torch.DoubleTensor(np.array(X[[x],:]))
        torchyhat = torchModel(xTensor)
        output = torchLoss(torchyhat, torch.DoubleTensor([[y[x]]]))

        model.backward(yhat, y[x])
        model.step()

        # print("output:", np.sum(np.abs(model.loss.forward(yhat, y[x]) - output.detach().numpy()))) 

        output.backward()
        optimizer.step()

        assert np.sum(np.abs(model.layers[0].W - torchModel[0].weight.T.detach().numpy())) < 1e-10, \
            "RealNN backward test failed, {} round linear 1 dW is different".format(i)
        assert np.sum(np.abs(model.layers[0].b - torchModel[0].bias.detach().numpy())) < 1e-10, \
            "RealNN backward test failed, {} round linear 1 db is different".format(i)
        assert np.sum(np.abs(model.layers[2].W - torchModel[2].weight.T.detach().numpy())) < 1e-10, \
            "RealNN backward test failed, {} round linear 2 dW is different".format(i)
        assert np.sum(np.abs(model.layers[2].b - torchModel[2].bias.detach().numpy())) < 1e-10, \
            "RealNN backward test failed, {} round linear 2 db is different".format(i)
    print("RealNN backward test passed")

if __name__ == "__main__":
    for i in range(10):
        print("Test {}:".format(i+1))
        
        random.seed(i+1)
        testLinear()
        testLinearBackward()
        testNN()
        testNNBackward()
        testSigmoid()
        testSigmoidBackward()
        mnist35Test()

