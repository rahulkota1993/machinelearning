import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

#print X.shape,y.shape,Xtest.shape,ytest.shape
#print X_i.shape

temp=np.linalg.inv(np.dot(X.transpose(),X))
print temp.shape
w=np.dot(temp,np.dot(X.transpose(),y))


print w.shape

temp2=np.linalg.inv(np.dot(X_i.transpose(),X_i))
print temp.shape
w2=np.dot(temp2,np.dot(X_i.transpose(),y))
print w2.shape
#test=(np.dot((y-np.dot(X,w)).transpose(),(y-np.dot(X,w)))/X.shape[0])**0.5
rmse=(np.dot((ytest-np.dot(Xtest,w)).transpose(),(ytest-np.dot(Xtest,w)))**0.5)/Xtest.shape[0]
rmse2=(np.dot((ytest-np.dot(Xtest_i,w2)).transpose(),(ytest-np.dot(Xtest_i,w2)))**0.5)/Xtest_i.shape[0]
print rmse.shape
print rmse2.shape
print rmse
print rmse2
#
#w = learnOLERegression(X,y)
#mle = testOLERegression(w,Xtest,ytest)
#
#w_i = learnOLERegression(X_i,y)
#mle_i = testOLERegression(w_i,Xtest_i,ytest)
#
#print('RMSE without intercept '+str(mle))
#print('RMSE with intercept '+str(mle_i))