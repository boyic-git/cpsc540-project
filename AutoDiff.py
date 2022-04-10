import numpy as np

class Ops:
    def __init__(self, inputs, gradients, outputs, operation):
        self.inputs = inputs
        self.gradients = gradients
        self.outputs = outputs
        self.operation = operation

class AutoDiff:
    def __init__(self):
        self.operations = []
        self.gradients = {}

    # inputs    -   list of input arrays
    # gradients -   list of gradient arrays corresponding to the input arrays
    def add(self, inputs, gradients, outputs, operation):
        self.operations.append(Ops(inputs, gradients, outputs, operation))
        # print(operation, len(inputs))
        for input in inputs:
            # print(operation)
            # print(input.shape)
            self.gradients[input.tobytes()] = np.zeros(input.shape)
            # print(input.shape)

    def backward(self, loss):
        for i in range(len(self.operations)-1, -1, -1):
            # print(i)
            op = self.operations[i]
            # loss = op.operation(loss, *op.inputs)
            if i == len(self.operations)-1:
                loss = op.operation(loss, *op.inputs)
            else:
                loss = op.operation(self.gradients[op.outputs.tobytes()], *op.inputs)
            for j in range(len(op.inputs)):
                self.gradients[op.inputs[j].tobytes()] += loss[j]
                if op.gradients[j] is not None:
                    op.gradients[j] += loss[j]
    
    def clear(self):
        self.gradients = {}
        self.operations = []
                    