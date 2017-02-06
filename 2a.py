import matplotlib.pyplot as plt
import numpy as np
import random

file = "/Users/sanjanakapistalam/Desktop/Deeplearning/iris-train.txt"
rows = open(file).read().splitlines()
#np.array(random.shuffle(rows)) #Shuffling the text file for each iteration                                                                                                       
#import pdb;pdb.set_trace()                                                                                                                                                      
#splitlines = [x.strip().split(' ') for x in rows]      
#cls = [x[0] for x in splitlines] #All classes    
#arr_cls = np.array(cls,dtype = np.uint8) #As an array                                                                                                                             
#f = [(x[1],x[2]) for x in splitlines] #All features                                                                                                                              
#arr_f = np.array(f,dtype=np.float32) #As an array   
D = 2 #Dimensionality
K = 3 #No of classes/categories


#Randomly initializing parameters
W = 0.01 * np.random.randn(D,K) #Weights
dW = 0
alpha = 0.05
s_size = 1
L2 = 0.01
lr = 0.001
bsize = 10
all_loss = []
for num in xrange(1000):
    #import pdb;pdb.set_trace()
    np.array(random.shuffle(rows)) #Shuffling the text file for each iteration
    #import pdb;pdb.set_trace()
    splitlines = [x.strip().split(' ') for x in rows]
    import pdb;pdb.set_trace()  
    cls = [x[0] for x in splitlines] #All classes 
    arr_cls = np.array(cls,dtype = np.uint8) #As an array 
    f = [(x[1],x[2]) for x in splitlines] #All features
    arr_f = np.array(f,dtype=np.float32) #As an array  
    for ix in xrange(bsize-1):
        int_cls = np.zeros((bsize,K),dtype = np.uint8)	
        #if ix==8:
           #import pdb;pdb.set_trace()
        start = (ix*bsize)
        stop = ((ix+1)*10)
        if np.any(ix!=0):
            scores = np.dot(arr_f[start:stop,[0,1]],W)
            for ind,i in enumerate(arr_cls[start:stop]):
                int_cls[ind,i-1] = 1
        else:
    	    scores = np.dot(arr_f[ix:bsize,[0,1]],W)
            for ind,i in enumerate(arr_cls[ix:bsize]):
                #import pdb;pdb.set_trace()
                int_cls[ind,i-1] = 1
        t_cls = int_cls
        scores -= np.max(scores)
        out = np.exp(scores)
        probs = out / np.sum(out)
        l_h = probs * t_cls
        #import pdb;pdb.set_trace()
        l_h[np.where(l_h==0.0000)] = 0.4
        loss = -np.log(l_h)
        #import pdb;pdb.set_trace()
        if np.any(ix!=0):
            diff_loss = np.dot(arr_f[start:stop].T,probs-t_cls)
        else:
            diff_loss =np.dot(arr_f[ix:bsize].T,probs-t_cls)
        dW = (alpha*dW) + (lr*diff_loss)
        W  = W - dW
        rloss = 0.5*L2*np.sum(W*W)
        tdiffloss = np.sum(diff_loss)/bsize
        tloss = (np.sum(loss)/bsize) +rloss
        #dW = (alpha*dW) + (lr*diff_loss)
        #W  = W -dW
    all_loss.append(tloss) 
    if 0%10 == 0:
      print "iteration %d: loss %f" % (num,tloss)
    scores_n = np.dot(arr_f,W)
    pred_cls= np.argmax(scores_n,axis=1)
    print 'training accuracy: %.2f' % (np.mean(pred_cls == arr_cls))
#------plots----------

#plt.subplot(1,2,1)
list = range(1000)
import pdb;pdb.set_trace()
plt.plot(list,all_loss,'-o')
plt.show()

