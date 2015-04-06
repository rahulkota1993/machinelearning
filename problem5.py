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
    
    
def learnRidgeERegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD   
    w=np.dot(np.dot(np.linalg.inv(lambd*X.shape[0]*np.identity(X.transpose().shape[0])+np.dot(X.transpose(),X)),X.transpose()),y)                                                
    return w
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    
#    w=w.reshape(65,1)
#    
#    
#
#    error=((np.dot((y-np.dot(X,w)).transpose(),(y-np.dot(X,w))))/(2*float(X.shape[0])))+ float(lambd*0.5*np.dot(w.transpose(),w))
#    #error_grad=((1/X.shape[0])*(np.dot(w.transpose(),np.dot(X.transpose(),X))-np.dot(y.transpose,X))+(lambd*w))
#    a=1/X.shape[0]
#    b=np.dot(w.transpose(),np.dot(X.transpose(),X))
#    c=np.dot(y.transpose(),X)
#    d=b-c
#    error_grad=(a*d)-(lambd*w)     
#sudeep   
    #wl=np.asmatrix(w)
    #wl=wl.transpose()
    wl=w.reshape(65,1)
    # error is the equation from problem 3
    # sudeep's error = ((np.dot((y-np.dot(X,wl)).transpose(),(y-np.dot(X,wl))))/(2*float(len(y))))+ (0.5*lambd*np.dot(wl.transpose(),wl))
    
    error=((0.5/X.shape[0])*np.dot((y-np.dot(X,wl)).transpose(),(y-np.dot(X,wl))))+(0.5*lambd*np.dot(wl.transpose(),wl))
     #error grad is the gradiance of the above equation
    

    error_grad =  ((np.dot(X.transpose(),(np.dot(X,wl)-y)))/float(len(y))) + (lambd*wl) 
    #grad=((1/X.shape[0])*(((-1)*(np.dot(y.transpose(),X)))+np.dot(wl.transpose(),np.dot(X.transpose,X))))+(lambd*wl) 
   
    #trial
    
   
  
    error_grad = np.squeeze(error_grad)
                                       
    return error, error_grad
    
    
    
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd=np.ones((x.shape[0],p+1))
    
    for i in range(x.shape[0]):
        for j in range(p+1):
            Xd[i][j]=x[i]**j
    return Xd

# Problem 5
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
k = 21
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
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeERegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeERegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.show()
plt.legend(('No Regularization','Regularization'))