import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pdb;

train_file = "./iris-train.txt" # "/Users/sanjanakapistalam/Desktop/Deeplearning/iris-train.txt"
test_file = "./iris-test.txt"
train_rows = open(train_file).read().splitlines()
test_rows = open(test_file).read().splitlines()
   
D = 2 #Dimensionality
K = 3 #No of classes/categories
	
#Randomly initializing parameters
W = 0.01 * np.random.randn(D,K) #Weights
dW = 0
alpha = 0.05
s_size = 1
L2 = 0.001
lr = 0.005
bsize = 10
all_tr_loss = []
all_tst_loss = []
train_res = []
test_res = []
for num in xrange(1000):
    np.array(random.shuffle(train_rows)) #Shuffling the train text file for each iteration
    splitlines = [x.strip().split(' ') for x in train_rows]
    train_cls = [x[0] for x in splitlines] #All classes 
    arr_traincls = np.array(train_cls,dtype = np.uint8) #As an array 
    trainf = [(x[1],x[2]) for x in splitlines] #All features
    arr_trainf = np.array(trainf,dtype=np.float32) #As an array
    np.array(random.shuffle(test_rows)) #Shuffling the test text file for each iteration
    splitlines_test = [x.strip().split(' ') for x in test_rows]
    test_cls = [x[0] for x in splitlines_test]
    arr_testcls = np.array(test_cls,dtype = np.uint8)
    testf = [(x[1],x[2]) for x in splitlines_test]
    arr_testf = np.array(testf,dtype=np.float32)

    for ix in xrange(bsize-1):
        int_tr_cls = np.zeros((bsize,K),dtype = np.uint8)
        int_tst_cls = np.zeros((bsize,K),dtype = np.uint8)	
        start = (ix*bsize)
        stop = ((ix+1)*10)
        stop = min(stop,arr_trainf.size)
        #import pdb;pdb.set_trace()
        tr_scores = np.dot(arr_trainf[start:stop,[0,1]],W)
        for ind,i in enumerate(arr_traincls[start:stop]):
            int_tr_cls[ind,i-1] = 1
               
        t_cls = int_tr_cls
        tr_scores -= np.max(tr_scores)
        out = np.exp(tr_scores)
        sum_  = out.sum(axis=1)
        probs = out / sum_[:, np.newaxis]
        #import pdb;pdb.set_trace()
        tr_loss = -np.log( np.max(probs,0.0000001) ) * t_cls
        diff_loss = -np.dot(arr_trainf[start:stop].T,t_cls-probs)

        rloss = 0.5*L2*np.sum(W*W)
        tloss = (np.sum(tr_loss)/bsize) + rloss
        
        dW = (alpha*dW) + (lr*diff_loss) #(L2*W) + (diff_loss)
        W  = W - dW #W - lr * dW    
    all_tr_loss.append(tloss) 
    int_tst_cls = np.zeros((len(arr_testf),K),dtype = np.uint8)
    tst_scores = np.dot(arr_testf[:,[0,1]],W)
    for index,val in enumerate(arr_testcls):
        int_tst_cls[index,val-1] = 1
    tst_cls = int_tst_cls
    tst_scores -= np.max(tst_scores)
    tst_out = np.exp(tst_scores)
    tst_sum_ = tst_out.sum(axis=1)
    tst_probs = tst_out / tst_sum_[:,np.newaxis]
    tst_loss = -np.log( np.max(tst_probs,0.0000001) ) * tst_cls
    t_tst_loss = (np.sum(tst_loss)/len(arr_testf)) + rloss
    all_tst_loss.append(t_tst_loss)
    if 0%10 == 0:
      print "iteration %d: train loss %f" % (num,tloss)
      print "iteration %d: test loss %f" % (num,t_tst_loss)
    import pdb;pdb.set_trace()
    scores_tr = np.dot(arr_trainf[:,[0,1]],W)
    pred_cls_tr= np.argmax(scores_tr,axis=1)+1
    train_acc = np.mean(pred_cls_tr == arr_traincls)
    train_res.append(train_acc)
    print 'training accuracy: %.2f' % (train_acc) #(np.mean(pred_cls == arr_traincls))
    scores_tst = np.dot(arr_testf[:,[0,1]],W)
    pred_cls_tst = np.argmax(scores_tst,axis=1)+1
    test_acc = np.mean(pred_cls_tst == arr_testcls)
    import pdb;pdb.set_trace()
    test_res.append(test_acc)
    print 'test accuracy: %.2f' % (test_acc)
    time.sleep(0.1)
#--------------Decision Boundaries(2b)-------
#import pdb;pdb.set_trace()
up_W = W
h = .02
X = arr_trainf[:,:2]
Y = arr_traincls
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
Data = arr_trainf[:,:2]
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
arr = np.array([xx.ravel(),yy.ravel()])
Score = np.dot(arr.T,up_W)
Score -= np.max(Score)
out1 = np.exp(Score)
sum1_  = out1.sum(axis=1)
prob = out1 / sum1_[:, np.newaxis]
Z = np.argmax(prob,axis=1)+1
Z = Z.reshape(xx.shape)
plt.figure(1,figsize=(4,3))
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='k',alpha=0.8,cmap=plt.cm.Paired)
plt.xlabel('Feature1(X1)')
plt.ylabel('Feature2(X2)')
plt.title('Decision Boundaries with scattered Training data points')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.axis("tight")
plt.show()

#------loss plots (2a)----------

plt.subplot(1,2,1)
list = range(1000)
#import pdb;pdb.set_trace()
plt.plot(list,all_tr_loss,'-',color='r',label='Training Loss')
plt.plot(list,all_tst_loss,'-',color='b',label='Test Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.subplot(1,2,2)
plt.plot(list,train_res,'-',color='r',label='Training Mean per class Accuracy')
plt.plot(list,test_res,'-',color='b',label='Test Mean per class Accuracy')
plt.title('Epoch vs Mean per class accuracy')
plt.xlabel('Epochs')
plt.ylabel('Mean per class accuracy')
plt.legend(loc='best')
plt.show()

