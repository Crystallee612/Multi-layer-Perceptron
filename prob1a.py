# Student: Xiaoting Li, prob 1a
import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1a: Softmax Regression \& the XOR Problem

@author - Alexander G. Ororbia II
'''

def computeNumGrad(X,y,theta,reg): # returns approximate nabla
	# WRITEME: write your code here to complete the routine
	eps = 1e-5
	# theta_list = list(theta)
	nabla_n = []
	nabla_param = [] 
	# NOTE: you do not have to use any of the code here in your implementation...

	# theta[0] and theta[1]
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
	# WRITEME: write your code here to complete the routine
	W = theta[0]
	b = theta[1]
	length = np.shape(X)[0]
	Fx = np.dot(X, W) + b
	f = np.array([np.exp(Fi) for Fi in Fx])
	VolumnSum = np.sum(f[:,i] for i in range(K))
	Pk = np.transpose(np.array([f[:,j]/VolumnSum for j in range(K)]))

	for i in range(length):
		Pk[i,y[i]] -= 1

	dW = np.dot(np.transpose(X), 1.0/length * Pk) + reg * W 
	db = 1.0/length * np.sum(Pk[i,:] for i in range(length))
	# dW = W * 0.0
	# db = b * 0.0	
	return [dW,db]


def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	length = np.shape(X)[0]
	Fx = np.dot(X, theta[0])+theta[1]
	f = np.array([np.exp(Fi) for Fi in Fx]) 
	VolumnSum = np.sum(f[:,i] for i in range(K))

	Py = np.array([f[i,y[i]]/VolumnSum[i] for i in range(length)])

	Cost = -1.0/length * np.sum(np.log(Py)) + reg/2 * np.sum(theta[0] ** 2)
	return Cost

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	W = theta[0]
	b = theta[1]
	length = np.shape(X)[0]
	# evaluate class scores
	scores = np.dot(X,W) + b
	# compute the class probabilities
	f = np.array([np.exp(si) for si in scores])
	VolumnSum = np.sum(f[:,i] for i in range(K))
	probs = np.array([f[i,:]/VolumnSum[i] for i in range(length)])

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

#Train a Linear Classifier

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters in such a way to play nicely with the gradient-check!
# W and b are consistent with X
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K)) + 1.0
theta = [W,b]

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

# Re-initialize parameters for generic training
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))
theta = [W,b]

n_e = 101
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 0.0 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(n_e):
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	# loss = 0.0
	loss = computeCost(X,y,theta,reg)
	if i % check == 0:
		print "iteration %d: loss %f" % (i, loss)

	# perform a parameter update
	# WRITEME: write your update rule(s) here
	d = computeGrad(X,y,theta,reg)
	# d = computeNumGrad(X,y,theta,reg)
	theta[0] -= step_size * d[0]
	theta[1] -= step_size * d[1] 

 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
# sys.exit(0) 

# evaluate training set accuracy
scores, probs = predict(X,theta)
#scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
# print (predicted_class)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))