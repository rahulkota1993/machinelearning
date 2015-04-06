import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle


X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/sample.pickle','rb')) 
groups=np.unique(y)
"""a=y.shape[0]
print a
print y.shape[0]
#Creating empty matrices for grouping two columns for classes in y
emp=[[] for i in range(X.shape[1]*len(groups))]
mtotal,mtotal1=[],[]

#adding elements to arrays by comparing features of y
for i in range(a):
    for j in range(len(groups)):
        if(y[i]==groups[j]):
            for k in range(X.shape[1]):
                emp[j+(k*len(groups))].append(X[i][k])
            
            
            
#for i in range(2*len(groups)):
   # print np.mean(emp[i])
   # print np.cov(emp[i])

meansarray=[]
for i in range(X.shape[1]*len(groups)):
    meansarray.append(np.mean(emp[i]))
    
    
#converting list to array
#print np.array(meanscov)
k=np.array(meansarray)

c=k.reshape(2,5)
d=c.transpose()
#print d

meantot=np.mean(X,axis=0)
#print meantot


#print c

#print c.shape"""
k=[]
k1=[]
k2=[]
for i in groups:
    temp=X[np.where(y==i)[0],]
    k.append(np.mean(temp,0))
    k1.append(temp)
    print temp.shape
ark=np.array(k)#means
ark1=np.array(k1)#arrays with same features

print ark[0].shape 
print ark1.shape


for i in groups:
    temp=X[np.where(y==i)[0],]
    cm=temp-np.mean(X,0)
    k2.append(np.dot(cm.transpose(),cm)/len(cm))

ark2=np.array(k2)#ark2=covariance
print ark2[0].shape

k3=np.zeros((2,2))
for i in range(len(groups)):
    k3=k3+((ark2[i]*ark1[i].shape[0]/X.shape[0]))
    
    

ark3=np.array(k3)#final covariance matrix
print ark[0].shape


    
iark3=np.linalg.inv(ark3)#inverse of final covariance matrix
print iark3

f=[]
mul=[]

#ark[0].reshape(1,2)*iark3*X.transpose()
for i in range(len(groups)):
    tempd=ark[i].reshape(1,2)
    mul=(np.dot(np.dot(tempd,iark3),Xtest.transpose()))-(0.5*np.dot(np.dot(tempd,iark3),tempd.transpose()))
    #mul=((tempd.dot(iark3).dot(Xtest.transpose()))-(0.5*temp.dot(iark3).dot(tempd.transpose())))
    f.append(mul)
    print mul.shape

arkf=np.array(f)
print arkf.shape
arkf1=arkf.reshape(5,100).transpose()
print arkf1.shape
label=np.argmax(arkf1,1)

label=label+1
label=label.reshape(100,1)
print label.shape

print('\n Training set Accuracy:' + str(100*np.mean((label == ytest).astype(float))) + '%')
#
#f1=[]
#mul1=[]
#
#for i in range(len(groups)):
#    tempd=ark[i].reshape(1,2)
#    iark3=np.linalg.inv(ark2[i])
#    print iark3
#    mul=-.5*np.log( np.linalg.det(iark3))-.5*(Xtest.transpose()-tempd.transpose())*iark3*(Xtest-tempd) 
#    #mul=(np.dot(np.dot(tempd,iark3),Xtest.transpose()))-(0.5*np.dot(np.dot(tempd,iark3),tempd.transpose()))
#    #mul=((tempd.dot(iark3).dot(Xtest.transpose()))-(0.5*temp.dot(iark3).dot(tempd.transpose())))
#    f1.append(mul)
#    print mul.shape
#    
#arkf2=np.array(f1)
#
#arkf2=arkf2.reshape(5,100).transpose()
#print arkf2.shape
##print arkf2
#
#label=np.argmax(arkf2,1)
#
#label=label+1
#label=label.reshape(100,1)
#print label.shape
#
#print('\n Training set Accuracy:' + str(100*np.mean((label == ytest).astype(float))) + '%')
##    
##
##
##
##        
##
