import matplotlib.pyplot as plt
import numpy as np
import cPickle
import sys
import random
import pandas as pd


file1 = "./cifar-10-batches-py/data_batch_1"
f1_open = open(file1,'rb')
dict1 = cPickle.load(f1_open)
data1 = dict1['data']
classes1 = dict1['labels']
imgs_cls1 = np.array(classes1)

#----file2----
file2 = "./cifar-10-batches-py/data_batch_2"
f2_open = open(file2,'rb')
dict2 = cPickle.load(f2_open)
data2 = dict1['data']
classes2 = dict2['labels']
imgs_cls2 = np.array(classes2)

#----file3-----
file3 = "./cifar-10-batches-py/data_batch_3"
f3_open = open(file3,'rb')
dict3 = cPickle.load(f3_open)
data3 = dict3['data']
classes3 = dict3['labels']
imgs_cls3 = np.array(classes3)

#----file4-----
file4 = "./cifar-10-batches-py/data_batch_4"
f4_open = open(file4,'rb')
dict4 = cPickle.load(f4_open)
data4 = dict4['data']
classes4 = dict4['labels']
imgs_cls4 = np.array(classes4)

#-----file5------
file5 = "./cifar-10-batches-py/data_batch_5"
f5_open = open(file5,'rb')
dict5 = cPickle.load(f5_open)
data5 = dict5['data']
classes5 = dict5['labels']
imgs_cls5 = np.array(classes5)

#--------------Test file --------
test_file = "./cifar-10-batches-py/test_batch"
tf_open = open(test_file,'rb')
test_dict = cPickle.load(tf_open)
test_data = test_dict['data']
test_cls = test_dict['labels']
tst_img_cls = np.array(test_cls)

#-----Implementing Softmax------
D = 3072 #Dimensionality                                                                                                                                                           
K = 10 #No of classes/categories 
                                                                                                                                                  
#-----Randomly initializing parameters
W = 0.01 * np.random.randn(D,K) #Weights  
dW = 0
alpha = 0.5
s_size = 1
L2 = 0.01#0.00001
lr = 0.001#1e-05
bsize = 100
all_tr_loss = []
all_tst_loss = []
train_res = []
pred_cls = []
all_images = np.vstack((data1,data2,data3,data4,data5))
all_classes = np.hstack((imgs_cls1,imgs_cls2,imgs_cls3,imgs_cls4,imgs_cls5))
for num in xrange(1000):
    all_list = list(zip(all_images,all_classes))
    random.shuffle(all_list)
    all_images,all_classes = zip(*all_list)
    all_images = np.array(all_images)
    all_classes = np.array(all_classes)
    test_list = list(zip(test_data,test_cls))
    random.shuffle(test_list)
    test_data,test_cls = zip(*test_list)
    test_data = np.array(test_data)
    test_cls = np.array(test_cls)
    for imgs in xrange(bsize-1):    
        class_arr = np.zeros((bsize,K),dtype = np.uint8)
        start = (imgs*bsize)
        stop = ((imgs+1)*bsize)
        stop = min(stop,all_images.size)
        X = all_images[start:stop]
        X = X/255 #normalizing
        
        score = np.dot(X[:,:],W)
        for ind,i in enumerate(all_classes[start:stop]):
            class_arr[ind,i-1] = 1
        #import pdb;pdb.set_trace()
        all_cls = class_arr
        score -= np.max(score)
        out = np.exp(score)
        sum_  = out.sum(axis=1)
        probs = out / sum_[:, np.newaxis]
        tr_loss = -np.log( np.max(probs,0.0000001) ) * all_cls
        diff_loss = -np.dot(X.T,all_cls-probs)
        rloss = 0.5*L2*np.sum(W*W)
        tloss = (np.sum(tr_loss)/bsize) + rloss
            
        dW = (alpha*dW) + (lr*diff_loss)#Implementing momentum
        W  = W - dW#Updating initialized weights                  
    all_tr_loss.append(tloss)
    if 0%100 == 0:
        print "iteration %d: train loss %f" % (num,tloss)
    
    scores_tr = np.dot(all_images,W)
    pred_cls_tr= np.argmax(scores_tr,axis=1)+1
    train_acc = np.mean(pred_cls_tr == all_classes)#all_classes[img_arr][start:stop])
    train_res.append(train_acc)
    print 'training accuracy: %.2f' % (train_acc)        
    
    scores_tst = np.dot((test_data[:,:])/255,W)
    pred_cls_tst = np.argmax(scores_tst,axis=1)+1
    pred_cls.append(pred_cls_tst)
    test_acc = np.mean(pred_cls_tst == tst_img_cls)
    print 'test accuracy: %.2f' % (test_acc)
#------Loss plot---------
list = range(1000)
plt.figure()                                                     
plt.plot(list,all_tr_loss,'-',color='r',label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Training loss vs Epoch')
plt.show()
#-------------Confusion matrix
#import pdb;pdb.set_trace()
act_cls = test_cls
y_actu = pd.Series(act_cls, name='Actual')
y_pred = pd.Series(pred_cls[0][:].tolist(), name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
plt.matshow(df_conf_norm)
plt.colorbar()
plt.show()
