#Student: Xiaoting Li, Problem #1d
import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1d: MLPs \& the Spiral Problem

@author - Alexander G. Ororbia II
'''

def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	length = np.shape(X)[0]

	hpre = np.dot(X, W) + b
	hpre = hpre * (hpre > 0)
	# for i in range(np.shape(hpre)[0]):
	# 	for j in range(np.shape(hpre)[1]):
	# 		if hpre[i,j]< 0:
	# 			hpre[i,j] = 0


	f = np.dot(hpre, W2) + b2
	F = np.array([np.exp(fi) for fi in f])
	VolumnSum = np.sum(F[:,i] for i in range(K))
	Py = np.array([F[i,y[i]]/VolumnSum[i] for i in range(length)])

	Cost = -1.0/length * np.sum(np.log(Py)) + reg/2 * (np.sum(W ** 2) + np.sum(W2 **2))
	return Cost		
			
def computeGrad(X,y,theta,reg): # returns nabla
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	# WRITEME: write your code here to complete the routine
	hpre = np.dot(X, W) + b
	hpre = hpre * (hpre > 0)

	length = np.shape(X)[0]
	f = np.dot(hpre, W2) + b2
	F = np.array([np.exp(fi) for fi in f])
	VolumnSum = np.sum(F[:,i] for i in range(K))
	Py = np.array([F[i,:]/VolumnSum[i] for i in range(length)])
	for j in range(length):
		Py[j,y[j]] -= 1

	dhpre = np.dot(1.0/length * Py, np.transpose(W2))

	dMid = dhpre * (hpre > 0)
	dW = np.dot(np.transpose(X), dMid) + reg * W
	db = np.array([np.sum(dMid[i,:] for i in range(length))])
	dW2 = np.dot(np.transpose(hpre), 1.0/length * Py) + reg * W2 
	db2 = np.array([1.0/length * np.sum(Py[i,:] for i in range(length))])
	return [dW,db,dW2,db2]

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	length = np.shape(X)[0]
	hpre = np.dot(X, W) + b 
	hpre = hpre * (hpre > 0)

	scores = np.dot(hpre, W2) + b2
	F = np.array([np.exp(fi) for fi in scores])
	VolumnSum = np.sum(F[:,i] for i in range(K))
	probs = np.array([F[i,:]/VolumnSum[i] for i in range(length)])

	return (scores,probs)
	
np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/spiral_train.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
theta = [W,b,W2,b2]

# some hyperparameters
n_e = 1001
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
for i in xrange(n_e):
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	loss = computeCost(X,y,theta,reg)
	if i % check == 0:
		print "iteration %d: loss %f" % (i, loss)

	# perform a parameter update
	# WRITEME: write your update rule(s) here
	d = computeGrad(X,y,theta,reg)
	# d = computeNumGrad(X,y,theta,reg)
	theta[0] -= step_size * d[0]
	theta[1] -= step_size * d[1]
	theta[2] -= step_size * d[2]
	theta[3] -= step_size * d[3]
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
# sys.exit(0) 
  

scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
					 
Z, P = predict(np.c_[xx.ravel(), yy.ravel()], theta)

Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.savefig(os.getcwd() + '/out/spiral_net.png')


plt.show()