import hdf5plugin
import h5py
import numpy as np

# f = h5py.File("mnist35.jld", "r")
f = h5py.File("basisData.jld", "r")
# (X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
X = f["X"][:]
y = f["y"][:]

with open('basisData.npz', 'wb') as f:
    # np.savez(f, X=X.T, y=y.T, Xtest=Xtest.T, ytest=ytest.T)
    np.savez(f, X=X.T, y=y.T)