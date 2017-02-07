import struct
import numpy as np
import scipy.io as sio
from scipy import linalg
import matplotlib.pyplot as plt

#---Function implementing PCA using SVD
#---Returns top k principal components and d-dimensional mean
def pca_svd(D,k):
    mn = np.mean(D,0)
    mean = np.zeros((60000,28,28))
    for i in xrange(784):
        mean[i,:,:] = mn
    X = D - mean
    X = np.reshape(X,(60000,784))
    Sigma = np.dot(X.T,X)
    U,S,V = np.linalg.svd(Sigma)
    PC = U[:,0:k]
    dM = np.mean(mn)
    return PC,dM
