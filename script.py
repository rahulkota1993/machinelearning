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
                    k1=[]
                    k2=[]
                    for i in groups:
                        temp=X[np.where(y==i)[0],]
                        k.append(np.mean(temp,0))
                        k1.append(temp)
                        tt=np.mean(X,0)
                        cm=temp-tt
                        k2.append(np.dot(cm.transpose(),cm)/len(cm))
                        
                    ark=np.array(k)#means
                    ark1=np.array(k1)#arrays with same features                
                    ark2=np.array(k2)#ark2=covariance matrices
                    
                    #Calculating covariance matrix combining all covariance matrices
                    k3=np.zeros((2,2))#Initialize a 2*2 Matrix
                    for i in range(len(groups)):
                        k3=(k3+(ark2[i]*ark1[i].shape[0]/X.shape[0]))#Covariance formula=
            
                    ark3=np.array(k3)#final covariance matrix    
                    iark3=np.linalg.inv(ark3)#inverse of final covariance matrix
                
                    means=ark
                    covmat=ark3
                    #print means
                    #print covmat.shape                    
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
    k1=[]
    k2=[]
    for i in groups:
        temp=X[np.where(y==i)[0],]
        k.append(np.mean(temp,0))
        k1.append(temp)
        tt=np.mean(X,0)
        cm=temp-tt
        k2.append(np.dot(cm.transpose(),cm)/len(cm))
    ark=np.array(k)#means
    ark1=np.array(k1)#arrays with same features
    ark2=np.array(k2)#ark2=covariance
    covmats=ark2
    means=ark
   
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    f=[]
    mul=[]
    groups=np.unique(ytest)#no. of classes which are unique
    iark3=np.linalg.inv(covmat)#inverse of final covariance matrix
    
    #ark[0].reshape(1,2)*iark3*X.transpose()
    for i in range(len(groups)):
        tempd=means[i].reshape(1,2)
       # print means[i]
        #print tempd
        mul=(np.dot(np.dot(tempd,iark3),Xtest.transpose()))-(0.5*np.dot(np.dot(tempd,iark3),tempd.transpose()))
        #mul=((tempd.dot(iark3).dot(Xtest.transpose()))-(0.5*temp.dot(iark3).dot(tempd.transpose())))
        f.append(mul)
        
    
    label=(np.argmax(np.array(f).reshape(5,100).transpose(),1)+1).reshape(100,1)
    #arkf=np.array(f)
    #arkf1=arkf.reshape(5,100).transpose()
    
    #label=np.argmax(arkf1,1)
    #
    #label=label+1
    #label=label.reshape(100,1)
    print('\n Training set Accuracy:' + str(100*np.mean((label == ytest).astype(float))) + '%')
    acc=100*np.mean((label == ytest).astype(float))
  
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    f=[]
    mul=[]
    groups=np.unique(ytest)
    
       
        
    for i in range(len(groups)):
        iark3=np.linalg.inv(covmats[i])
        kk=np.linalg.det(covmats[i])
        iark3_det=np.log(kk)
        tempd=means[i]
        tempd=tempd.reshape(1,2)       
        
        for j in range(len(Xtest)):
                 
            a=Xtest[j]
            b=means[i]
            c=a-b
            d=c.transpose()
            f1=iark3_det
            f2=np.dot(d,iark3)
            f3=np.dot(f2,c)
            fi=(-0.5)*(f1+f3) 
            f.append(fi)
            
    label=(np.argmax(np.array(f).reshape(5,100).transpose(),1)+1).reshape(100,1)
#from here
#arkf=arkf.reshape(5,100).transpose()
#label=np.argmax(arkf,1)
#
#label=label+1
#label=label.reshape(100,1)
#print label.shape
##from here
#    f=[]
#    mul=[]
#    groups=np.unique(ytest)
#    #ark[0].reshape(1,2)*iark3*X.transpose()
#    for i in range(len(groups)):
#        iark3=np.linalg.inv(covmats[i])
#        iark3_det=np.log(np.linalg.det(covmats[i]))
#        tempd=means[i].reshape(1,2)
#        
#        
#        for j in range(len(Xtest)):
#                
#            a=Xtest[j]-means[i]
#            b=a.transpose()
#            
#            fi=(-0.5)*iark3_det- (0.5)*np.dot(np.dot(b,iark3),a)
#    
#            f.append(fi)
#    
#    
#    arkf=np.array(f)
#    print arkf.shape
#    
#    arkf=arkf.reshape(5,100).transpose()
#    print arkf.shape
#    #
#    #arkf1=arkf.reshape(5,100).transpose()
#    #print arkf1
#    label=np.argmax(arkf,1)
#    
#    label=label+1
#    label=label.reshape(100,1)
#    print label.shape
#    
#    print('\n Training set Accuracy:' + str(100*np.mean((label == ytest).astype(float))) + '%')
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
    #rmse=(np.dot((ytest-np.dot(Xtest,w)).transpose(),(ytest-np.dot(Xtest,w)))**0.5)/Xtest.shape[0]
    
    rmse=np.sqrt(np.dot((ytest-np.dot(Xtest,w)).transpose(),(ytest-np.dot(Xtest,w))))/Xtest.shape[0]
    
    return rmse

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

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('C:/Users/Rahul/Desktop/python2/sample.pickle','rb') )           

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
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
mle = testOLERegression(w,Xtest,ytest)
print w.shape

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l =learnRidgeERegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)


# Problem 4
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
plt.plot(lambdas,rmses4)

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
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
