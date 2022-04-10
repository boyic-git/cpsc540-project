import hdf5plugin
import h5py
import numpy as np

f = h5py.File("mnist35.jld", "r")
X = f["X"][:]
y = f["y"][:]
y[y==2] = -1
Xtest = f["Xtest"][:]
ytest = f["ytest"][:]
ytest[ytest==2] = -1

with open('mnist35.npz', 'wb') as f:
    np.savez(f, X=X.T, y=y.T, Xtest=Xtest.T, ytest=ytest.T)