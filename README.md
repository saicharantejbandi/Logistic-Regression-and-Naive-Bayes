# Logistic-Regression-and-Naive-Bayes
## Performed Classification task of MNIST data (two numbers 7,8) using two approaches from scratch
  a. Naive Bayes
  b. Logistic Regressions
  
 ## Naive Bayes 
Two features are extracted from each image 
  1. Standard Deviation of all pixel values
  2. Mean of all Pixel Values
  
Using Naive Bayes->P( (Y=Digit 7) | X )=P(X | (Y=Digit 7))*P(Y=Digit 7) / P(X) 
P(X | (Y igit 7)) is the probability of finding in the distribution where they are classified as digit 7. 1 = D X1 X1
Which can be estimated by the Maximum Likelihood Equation for Normal Distribution.

Similarly for each class the probability is calculated and assigned a class which has higher probability. 

The Accuracy obtained from Naive Bayes is 62.73%.

## Logistic Regression

As already the features have been extracted, the features can be used to implement logistic regression. To implement logistic regression we use sigmoid/logistic function on the linear regression hypothesis to classify the inputs either 0(digit 7) or 1 (digit 8)

Yprediction=WTX  where W is weight/coefficient matrix (T represents Trasnpose) and X is the input feature matrix. Upon applying the sigmoid function to this we get the final hypothesis for the logistic regression. Where the sigmoid function is given by,
(t)=1/1+e-t. Now applying sigmoid to out linear regression hypothesis i.e,

Logistic Regression Hypothesis(h)=(WTX)=1/1+e-WTX

The W can be calculated Conditional by Log-Likelihood function. As we know this does not produce closed form solution, we can use gradient ascent of gradient descent to calculate W(weights).

The accuracy obtained from Logistic Regression is 84.90%.
The indepth explanation and formulae can be found in the report.pdf.

# The code for Naive Bayes implementation is in Naive Bayes.py
# The code for Logistsic Regression implementation is in Logistsic Regression.py















