import matplotlib.pyplot as plt
import numpy as np
import cPickle
import sys
file = "./cifar-10-batches-py/data_batch_1"
f_open = open(file,'rb')
dict = cPickle.load(f_open)
img_data = dict['data']
classes = dict['labels']
img_cls = np.array(classes)
imgs = []
for i in range(3):
    for ind,val in enumerate(np.unique(classes)):
        cls_ind = np.where(img_cls == ind)
        im = img_data[cls_ind][i].reshape(3,32,32).transpose(1,2,0)
        imgs.append(im)
for img in range(30):
    plt.subplot(3,10,img+1)
    #import pdb;pdb.set_trace()
    plt.imshow(imgs[img])
    plt.axis('off')
plt.tight_layout()
plt.show()
