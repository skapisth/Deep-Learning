import os
import struct
import numpy as np
from load_mnist import *
import matplotlib.pyplot as plt

#---loads MNIST images----
#---Computes the mean and standar deviation----
#---first row of the plot will display meanof each digit---
#---Second row will display standard deviation----

train_img_file = './train-images-idx3-ubyte'
train_lbl_file = './train-labels-idx1-ubyte'
test_img_file  = './t10k-images-idx3-ubyte'
test_lbl_file  = './t10k-labels-idx1-ubyte'

with open(train_lbl_file, 'rb') as lbl_f:
    magic, num = struct.unpack(">II", lbl_f.read(8))
    lbl = np.fromfile(lbl_f, dtype = np.int8)
    
with open(train_img_file, 'rb') as img_f:
    magic, num, rows, cols = struct.unpack(">IIII", img_f.read(16))
    img = np.fromfile(img_f, dtype = np.int8).reshape(len(lbl), rows, cols)
    
import pdb;pdb.set_trace()
                                                                                                                             
for i in xrange(10):
    Img = img.T
    M = np.mean(img[(lbl==i),:,:],0)
    S = np.std(img[(lbl==i),:,:],0)
    plt.subplot(2,10,i+1)
    plt.imshow(M)
    plt.axis('off')
    plt.subplot(2,10,i+11)
    plt.imshow(S)
    plt.axis('off')
plt.show()
