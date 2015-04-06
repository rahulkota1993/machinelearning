import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle


X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/sample.pickle','rb')) 
groups=np.unique(y)
k=[]
k1=[]
k2=[]
for i in groups:
    temp=X[np.where(y==i)[0],]
    k.append(np.mean(temp,0))
    k1.append(temp)
    
ark=np.array(k)#means
ark1=np.array(k1)#arrays with same features

for i in groups:
    temp=X[np.where(y==i)[0],]
    cm=temp-np.mean(X,0)
    k2.append(np.dot(cm.transpose(),cm)/len(cm))

ark2=np.array(k2)#ark2=covariance
print ark2.shape


f=[]
mul=[]

#ark[0].reshape(1,2)*iark3*X.transpose()
for i in range(len(groups)):
    iark3=np.linalg.inv(ark2[i])
    iark3_det=np.log(np.linalg.det(ark2[i]))
    tempd=ark[i].reshape(1,2)
    
    
    for j in range(len(Xtest)):
            
        a=Xtest[j]-ark[i]
        b=a.transpose()
        
        fi=(-0.5)*(iark3_det+np.dot(np.dot(b,iark3),a))

        f.append(fi)
   

arkf=np.array(f)
print arkf.shape

arkf=arkf.reshape(5,100).transpose()
print arkf.shape
#
#arkf1=arkf.reshape(5,100).transpose()
#print arkf1
label=np.argmax(arkf,1)

label=label+1
label=label.reshape(100,1)
print label.shape

print('\n Training set Accuracy:' + str(100*np.mean((label == ytest).astype(float))) + '%')
