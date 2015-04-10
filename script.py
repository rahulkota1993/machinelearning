# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
                    groups=np.unique(y)
                    k=[]
                    for i in groups:
                        temp=X[np.where(y==i)[0],]
                        k.append(np.mean(temp,0))
                       
                        
                    ark=np.array(k)#means               
                    means=ark.transpose()
                    covmat=np.cov(X.transpose())
                    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    groups=np.unique(y)
    k=[]
    k2=[]
    for i in groups:
        temp=X[np.where(y==i)[0],]
        k.append(np.mean(temp,0))
        k2.append(np.cov(temp.transpose()))
    ark=np.array(k)#means
    ark2=np.array(k2)#ark2=covariance
    covmats=ark2
    means=ark.transpose()
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    groups=np.unique(ytest)
    iark3=np.linalg.inv(covmat)#inverse of final covariance matrix
    det_iark3=np.linalg.det(covmat)
    result=[]
    means=means.transpose()
    for i in range(ytest.shape[0]):
        temp=np.mat(Xtest[i,:])
        mul=[]
        for j in range(len(groups)):
            tempmeans=np.mat(means[j,:])
            tempresult=temp-tempmeans
            a=(1/((2*3.14*(det_iark3**2))**0.5))
            b=np.exp(-0.5*np.dot(np.dot(tempresult,iark3),tempresult.transpose()))
            mul.append(a*b)
        mul=np.array(mul)
        result.append(np.argmax(mul)+1)
    result=np.array(result)
    label=result.reshape(ytest.shape[0],1)     
      
            
        
    
   # print('\n Training set Accuracy:' + str(100*np.mean((label == ytest).astype(float))) + '%')
    acc=100*np.mean((label == ytest).astype(float))
  
    return acc,label

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    groups=np.unique(ytest)
    result=[]
    means=means.transpose()
    for i in range(ytest.shape[0]):
        temp=np.mat(Xtest[i,:])
        mul=[]
        for j in range(len(groups)):
            iark3=np.linalg.inv(covmats[j])#inverse of final covariance matrix
            det_iark3=np.linalg.det(covmats[j])
            tempmeans=np.mat(means[j,:])
            tempresult=temp-tempmeans
            a=(1/((2*3.14*(det_iark3**2))**0.5))
            b=np.exp(-0.5*np.dot(np.dot(tempresult,iark3),tempresult.transpose()))
            mul.append(a*b)
        mul=np.array(mul)
        result.append(np.argmax(mul)+1)
    result=np.array(result)
    label=result.reshape(ytest.shape[0],1)  
        
    acc=100*np.mean((label == ytest).astype(float))
    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD   
    temp=np.linalg.inv(np.dot(X.transpose(),X))
    
    w=np.dot(temp,np.dot(X.transpose(),y))   
    
                                            
    return w

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

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    rmse=(np.dot((ytest-np.dot(Xtest,w)).transpose(),(ytest-np.dot(Xtest,w)))**0.5)/Xtest.shape[0]
    
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    w=w.reshape(65,1)
   
    error=((0.5/X.shape[0])*np.dot((y-np.dot(X,w)).transpose(),(y-np.dot(X,w))))+(0.5*lambd*np.dot(w.transpose(),w))
    
    kk=((np.dot(w.transpose(),np.dot(X.transpose(),X))-np.dot(y.transpose(),X))/X.shape[0]).transpose()+ (lambd*w)
    
    error_grad=np.squeeze(kk)
    
                                
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

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/sample.pickle','rb') )           

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,we = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

#x1 = np.linspace(0,16,100)
#x2 = np.linspace(0,16,100)
#xv,yv = np.meshgrid(x1,x2)
#xx=np.zeros((x1.shape[0]*x2.shape[0],2))
#xx[:,0]=xv.ravel()
#xx[:,1]=yv.ravel()
#ldaacc1,lda_est_labels1 = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
##print lda_est_labels1.shape
#plt.contourf(x1,x2,lda_est_labels1.reshape((x1.shape[0],x2.shape[0])))
##print (lda_est_labels1.reshape((x1.shape[0],x2.shape[0]))).shape
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
#plt.show()

# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
plt.plot(w)
mle = testOLERegression(w,Xtest,ytest)
mle_training=testOLERegression(w,X,y)
w_i = learnOLERegression(X_i,y)
plt.plot(w_i)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_itraining=testOLERegression(w_i,X_i,y)
print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))
print('RMSE without intercept for training'+str(mle_training))
print('RMSE with intercept for training' + str(mle_itraining))
# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l =learnRidgeERegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
rms=np.zeros((k,1))
i=0
for lambd in lambdas:
    w_l =learnRidgeERegression(X_i,y,lambd)
    rms[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
#plt.plot(lambdas,rms)
plt.plot(w_l)
plt.show()


# Problem 4
k = 101
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
#plt.plot(lambdas,rmses4)
#plt.show()

#print rmses4




# Problem 5
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
#plt.plot(range(pmax),rmses5)
#plt.legend(('No Regularization','Regularization'))
