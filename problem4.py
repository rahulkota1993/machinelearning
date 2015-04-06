import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/diabetes.pickle','rb')) 

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
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD      
    
    error=((np.dot((y-np.dot(X,w)).transpose(),(y-np.dot(X,w))))/(2*X.shape[0]))+ (lambd*0.5*np.dot(w.transpose(),w))
    
    #error_grad=((1/X.shape[0])*(np.dot(w.transpose(),np.dot(X.transpose(),X))-np.dot(y.transpose,X))+(lambd*w))
    a=1/X.shape[0]
    b=np.dot(w.transpose(),np.dot(X.transpose(),X))
    c=np.dot(y.transpose(),X)
    d=b-c
    error_grad=(a*d)-(lambd*w)
    print error_grad.shape
    
    
    
    error_grad=0                
       
    return error, error_grad
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
k=21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1