import struct
import numpy as np
import scipy.io as sio
from scipy import linalg
import matplotlib.pyplot as plt
from PCA_SVD import pca_svd

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
#---Use the PCA_SVD function
P,D = pca_svd(im,10)
I = P.reshape(28,28,10)
max = np.max(I)
min = np.min(I)
N_I = (I-min)/(max-min) #Contrast stretching

#----Plot----
for num in xrange(10):
    plt.subplot(2,5,num+1)
    plt.imshow(N_I[:,:,num])
    plt.axis('off')
plt.show()
