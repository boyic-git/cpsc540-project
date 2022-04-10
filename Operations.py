import numpy as np

# a and b are all numpy array with any shape
# a + b
def add(grad, a, b):
    return grad, grad

# a - b
def sub(grad, a, b):
    return grad, -grad

# a * b
def mul(grad, a, b):
    return b*grad, a*grad

# a / b
def div(grad, a, b):
    # scalar: 1/b * grad
    da = np.divide(np.ones(b.shape), b)*grad
    # scalar: -a/(b**2) * grad
    db = np.divide(-a, b**2)*grad
    return da, db

# a matmul b: A*B
def matmul(grad, a, b):
    da = np.matmul(grad, b.T)
    db = np.matmul(a.T, grad)
    return da, db

def exp(grad, a):
    return [grad * np.exp(a)]

def log(grad, a):
    return [1/a * grad]
