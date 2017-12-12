#Student: Xiaoting Li  Problem #1c
import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1c: MLPs \& the XOR Problem

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

def computeNumGrad(X,y,theta,reg): # returns approximate nabla
	# WRITEME: write your code here to complete the routine
	eps = 1e-5
	# theta_list = list(theta)
	nabla_n = []
	nabla_param = []
	# NOTE: you do not have to use any of the code here in your implementation...
	for i in range(len(theta)):
		for j in range(len(theta[i])):
			for k in range(len(theta[i][j])):
				theta[i][j][k] += eps
				J_l = computeCost(X,y,theta,reg)
				theta[i][j][k] -= 2*eps
				J_r = computeCost(X,y,theta,reg)
				theta[i][j][k] += eps
				param_grad = (J_l - J_r) / (2 * eps)
				nabla_param.append(param_grad)

		nabla_param = np.reshape(nabla_param, np.shape(theta[i]))
		nabla_n.append(nabla_param)
		nabla_param = []

	return nabla_n			
			
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
	# print [dW, db, dW2, db2]
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
path = os.getcwd() + '/data/xor.dat'  
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

# initialize parameters in such a way to play nicely with the gradient-check! 
h = 6 #100 # size of hidden layer
W = 0.05 * np.random.randn(D,h) #0.01 * np.random.randn(D,h)
b = np.zeros((1,h)) + 1.0
W2 = 0.05 * np.random.randn(h,K) #0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K)) + 1.0
theta = [W,b,W2,b2] 

# some hyperparameters
reg = 1e-3 # regularization strength

nabla_n = computeNumGrad(X,y,theta,reg)
nabla = computeGrad(X,y,theta,reg)
nabla_n = list(nabla_n)
nabla = list(nabla)

for jj in range(0,len(nabla)):
	is_incorrect = 0 # set to false
	grad = nabla[jj]
	grad_n = nabla_n[jj]
	err = np.linalg.norm(grad_n - grad) / (np.linalg.norm(grad_n + grad))
	if(err > 1e-8):
		print("Param {0} is WRONG, error = {1}".format(jj, err))
	else:
		print("Param {0} is CORRECT, error = {1}".format(jj, err))

# re-init parameters
h = 6 #100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
theta = [W,b,W2,b2] 


# some hyperparameters
n_e = 101
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 0.0 # regularization strength
	
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