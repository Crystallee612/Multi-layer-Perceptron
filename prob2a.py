#Student: Xiaoting Li, problem 2a
from __future__ import print_function
import os 
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 2a: 1-Layer MLP for IRIS

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
	
def create_mini_batch(X, y, start, end):
	# WRITEME: write your code here to complete the routine
	mb_x = X[start: end]
	mb_y = y[start: end]
	return (mb_x, mb_y)
		
def shuffle(X,y):

	ii = np.arange(X.shape[0])
	ii = np.random.shuffle(ii)
	X_rand = X[ii]
	y_rand = y[ii]
	print (X_rand.shape[1:])
	X_rand = X_rand.reshape(X_rand.shape[1:])
	y_rand = y_rand.reshape(y_rand.shape[1:])
	return (X_rand,y_rand)
	
np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/iris_train.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

# load in validation-set
path = os.getcwd() + '/data/iris_test.dat'
data = pd.read_csv(path, header=None) 
cols = data.shape[1]  
X_v = data.iloc[:,0:cols-1]  
y_v = data.iloc[:,cols-1:cols] 

X_v = np.array(X_v.values)  
y_v = np.array(y_v.values)
y_v = y_v.flatten()


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
n_e = 2
n_b = 10
check = 10
step_size = 0.01 #1e-0
reg = 1e-3 #1e-3 # regularization strength

train_cost = []
valid_cost = []
# gradient descent loop
num_examples = X.shape[0]
for i in xrange(n_e):
	X, y = shuffle(X,y) # re-shuffle the data at epoch start to avoid correlations across mini-batches
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	train_cost.append(computeCost(X,y,theta,reg))
	valid_cost.append(computeCost(X_v,y_v,theta,reg))

	# WRITEME: write the inner training loop here (1 fullC pass, but via mini-batches instead of using the full batch to estimate the gradient)
	s = 0
	while (s < num_examples):
		# build mini-batch of samples
		X_mb, y_mb = create_mini_batch(X,y,s,s + n_b)
		
		# WRITEME: gradient calculations and update rules go here
		d_mb = computeGrad(X_mb, y_mb, theta, reg)
		theta[0] -= step_size * d_mb[0]
		theta[1] -= step_size * d_mb[1]
		theta[2] -= step_size * d_mb[2]
		theta[3] -= step_size * d_mb[3]
		s += n_b
# print (train_cost)
# print (valid_cost)
print(' > Training loop completed!')
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
sys.exit(0) 

scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: {0}'.format(np.mean(predicted_class == y)))

scores, probs = predict(X_v,theta)
predicted_class = np.argmax(scores, axis=1)
print('validation accuracy: {0}'.format(np.mean(predicted_class == y_v)))

# NOTE: write your plot generation code here (for example, using the "train_cost" and "valid_cost" list variables)

plt.plot(train_cost,'r--',label='train_cost')  
plt.plot(valid_cost,'b--',label='valid_cost')
plt.xlim(0,n_e)
plt.xlabel("epoch")
plt.ylabel("Cost")
plt.legend()
plt.savefig(os.getcwd() + '/out/mini_batch #1a')
plt.show()


