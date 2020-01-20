# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:36:48 2019

@author: saich
"""


import numpy as np


import scipy.io


Numpyfile= scipy.io.loadmat("mnist_data.mat") 

trX=Numpyfile['trX']


trY=Numpyfile['trY']

tsX=Numpyfile['tsX']

tsY=Numpyfile['tsY']

x1=trX.std(axis=1) #extracting feature1 from training data aka standard deviation
x1=x1.reshape(12116,1)
x2=trX.mean(axis=1).reshape(12116,1) #extracting feature2 from training data aka mean 

fX=np.concatenate((x1,x2),axis=1) #taking featues mean and standard deviation in to one matrix

x1=tsX.std(axis=1)  #extracting feature1 from testing data aka standard deviation
x1=x1.reshape(2002,1)
x2=tsX.mean(axis=1).reshape(2002,1)  #extracting feature2 from testing data aka standard deviation

fsX=np.concatenate((x1,x2),axis=1)

fX7=fX[0:6265] #saving features of digit 7 into one matrix

fX8=fX[6265:] #saving features of digit 8 into one matrix

fX7mean=fX7.mean(axis=0).reshape(1,2) #parameters mean for each distribution
fX7std=fX7.std(axis=0).reshape(1,2) #parameters std for each distribution

print("The means of feature 1 and feature 2 of digit 7 are ")
print(fX7mean)

print("The standard deviations of feature 1 and feature 2 of digit 7 are ")
print(fX7std)



fX8mean=fX8.mean(axis=0).reshape(1,2)    #parameters mean for each distribution
fX8std=fX8.std(axis=0).reshape(1,2)    #parameters std for each distribution

print("The means of feature 1 and feature 2 of digit 8 are ")
print(fX8mean)

print("The standard deviations of feature 1 and feature 2 of digit 8 are ")
print(fX8std)


p7=6265/12116 #finding probalability of y being classified as 7
print("The probablity of Digit 7: ", p7)
print("The probablity of Digit 8: ", 1-p7)

def probabilityND(x, mean, std):
    t=((x-mean)**2)/(2*std**2)
    exp=(np.e)**-t 
    return ((exp/(std*(2*np.pi)**0.5)))







pr_7=probabilityND(fsX,fX7mean,fX7std).prod(axis=1).reshape(2002,1) #finding the probabalility of given x in digit 7 distribution
pr_8=probabilityND(fsX,fX8mean,fX8std).prod(axis=1).reshape(2002,1) #finding the probabalility of given x in digit 8 distribution
pr_7=pr_7*p7 #finding the probabailty of x being classified as digit7
pr_8=pr_8*(1-p7) #finding the probabailty of x being classified as digit8

for i in range(0,2002):
    if pr_7[i]>pr_8[i]:
        pr_7[i]=0      #filling pr_7 with 0's of it is classified as 7
    else:
        pr_7[i]=1      #filling pr_7 with 1's of it is classified as 8


count=0
tsY=tsY.T #transposed to compare with the result

for i in range(tsY.shape[0]):
    if tsY[i]==pr_7[i]:
        
        count=count+1
    
print('Accuracy of Naive Bayes is:')
print(count/(tsY.shape[0])*100)

count=0

for i in range(0,1028):
    if tsY[i]==pr_7[i]:
        
        count=count+1

print('Accuracy of Naive Bayes classifying Digit 7 is:')
print((count/1028)*100)

count=0
for i in range(1028,2002):
    if tsY[i]==pr_7[i]:
        
        count=count+1

print('Accuracy of Naive Bayes classifying Digit 8 is:')
print((count/974)*100)