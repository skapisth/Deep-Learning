import struct
import numpy as np
import scipy.io as sio
from scipy import linalg
import matplotlib.pyplot as plt
from PCA_SVD import pca_svd
#------Plots Mean Reconstruction Error-----

#----loads MNIST data                                                           
train_img_file = './train-images-idx3-ubyte'
train_lbl_file = './train-labels-idx1-ubyte'
#----get in array format---                                                     
with open(train_lbl_file, 'rb') as lbl_f:
    magic, num = struct.unpack(">II", lbl_f.read(8))
    lbl = np.fromfile(lbl_f, dtype = np.int8)

with open(train_img_file, 'rb') as img_f:
    magic, num, rows, cols = struct.unpack(">IIII", img_f.read(16))
    img = np.fromfile(img_f, dtype = np.int8).reshape(len(lbl), rows, cols)

im = img

P,D,X = pca_svd(im,784)
Z = np.dot(X,P.T)
imR = np.dot(Z,P) # Reconstructing from reduced representation
imO = np.reshape(im,(60000,784))
abs = np.abs(imR - imO)
sq = np.square(abs)
MRE = np.mean(sq,0)
k = 784

#-----Plot----
plt.plot(np.arange(k),MRE)
plt.xlabel('k - No of Principal Components')
plt.ylabel('Error')
plt.title('Mean Reconstruction Error')
plt.show()
