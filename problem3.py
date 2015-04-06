import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/diabetes.pickle','rb'))  
def learnRidgeERegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD   
    w=np.dot(np.dot(np.linalg.inv(X.shape[0]*lambd*np.identity(X.transpose().shape[0])+np.dot(X.transpose(),X)),X.transpose()),y)                                                
    return w
def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    rmse=(np.dot((ytest-np.dot(Xtest,w)).transpose(),(ytest-np.dot(Xtest,w)))**0.5)/Xtest.shape[0]
   # print rmse
    return rmse  
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
k = 21
lambdas = np.linspace(0, 1.0, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l =learnRidgeERegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
plt.show()
#temp=np.dot(np.dot(np.linalg.inv(lambdas[0]*np.identity(X.transpose().shape[0])+np.dot(X.transpose(),X)),X.transpose()),y)
#print temp.shape