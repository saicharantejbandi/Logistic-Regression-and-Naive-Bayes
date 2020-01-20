# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:36:48 2019

@author: saich
"""


import numpy as np


import scipy.io

print(1)


Numpyfile= scipy.io.loadmat("mnist_data.mat") 

trX=Numpyfile['trX']




trY=Numpyfile['trY']




tsX=Numpyfile['tsX']





tsY=Numpyfile['tsY']




x1=trX.std(axis=1)
x1=x1.reshape(trX.shape[0],1)
x2=trX.mean(axis=1).reshape(trX.shape[0],1)

trX=np.concatenate((x1,x2),axis=1) #featues aka mean and standard deviation have been extracted for training data

x1=tsX.std(axis=1)
x1=x1.reshape(tsX.shape[0],1)
x2=tsX.mean(axis=1).reshape(tsX.shape[0],1)

tsX=np.concatenate((x1,x2),axis=1) #featues aka mean and standard deviation have been extracted for testing data




ones = np.ones((trX.shape[0], 1))  # adding ones as another feature for training data
trX= np.concatenate((ones, trX), axis=1)





ones = np.ones((tsX.shape[0], 1))
tsX= np.concatenate((ones, tsX), axis=1) #adding one as another feature for testing data


def hyp(x,w):
    return(np.dot(x,w))
    
def sigmoid(z):  #defining sigmoid function
    return(1/(1+np.exp(-z)))
    



def gradient(x,h,y):
    g=np.dot((x.T),(y.T-h))
    
    return g







w= np.zeros(trX.shape[1])





#w=w.reshape(785,1)
w=w.reshape(3,1)

trX.shape


alpha=0.001 #learning rate




for i in range(1000000): #implementing gradient ascent
    z=np.dot(trX,w)
    h=sigmoid(z)
    h=h.reshape(trY.shape[1],1)
    g=gradient(trX,h,trY)
    
    w=w+alpha*g
    

def predict(X, w, threshold=0.5):   #to classify the output depending on the threshold value
    return sigmoid(hyp(X, w)) >= threshold #gives the final prediction


re=predict(tsX,w) #final predictions
count=0
tsY=tsY.T #transposed to compare with the result 

print("The values of Parameters are")
print(w)
for i in range(tsY.shape[0]):
    if tsY[i]==re[i]:
        
        count=count+1
    
print('Accuracy of Logistic Regression is:')
print(count/(tsY.shape[0])*100)


count=0

for i in range(0,1028):
    if tsY[i]==re[i]:
        
        count=count+1

print('Accuracy of Logistic Regression classifying Digit 7 is:')
print((count/1028)*100)

count=0
for i in range(1028,2002):
    if tsY[i]==re[i]:
        
        count=count+1

print('Accuracy of Logistic Regression classifying Digit 8 is:')
print((count/974)*100)


