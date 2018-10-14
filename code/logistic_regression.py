import numpy as np
from math import exp
import random
import matplotlib.pyplot as plt

TRAINING_FILE_PATH = 'credit.txt'
DEGREE = 3
LAMBD = 0.1

# Partitioning the data into training and testing
def data_partition(data,frac):
	n = len(data)
	training_data_n = round(frac*n)
	r=random.sample(range(0, n), n)
	training_data = []
	validation_data = []
	for i in range(training_data_n):
		training_data.append(data[r[i]])

	for i in range(training_data_n,n):
		validation_data.append(data[r[i]])

	return training_data,validation_data

def get_X_Y(data):
	n = len(data)
	x0 = np.ones(n)
	x1 = []
	x2 = []
	Y = []

	for i in range(n):
		x1.append(data[i][0])
		x2.append(data[i][1])
		Y.append(data[i][2])

	x1 = np.array(x1).astype(np.float)
	x2 = np.array(x2).astype(np.float)

	X = np.array([x0,x1,x2]).T
	Y = np.array(Y).astype(np.float)

	return X,Y


def plot_data(data):
	n = len(data)
	x1_yes = []
	x2_yes = []
	x1_no = []
	x2_no = []

	for i in range(n):
		if data[i][2]=='1':
			x1_yes.append(data[i][0])
			x2_yes.append(data[i][1])
		elif data[i][2]=='0':
			x1_no.append(data[i][0])
			x2_no.append(data[i][1])

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(x1_no, x2_no, s=60, c='red',label='Rejected')
	ax1.scatter(x1_yes, x2_yes, s=60, c='blue',label='Accepted')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('Credit Card Application')
	plt.legend(loc='upper left')
	plt.show()


# Data reading from file
file = open(TRAINING_FILE_PATH,'r')
data = file.read().split('\n')
file.close()
if '' in data:
	data.remove('')

n=len(data)

for i in range(n):
	data[i]=data[i].split(',')


def cost_function(X, Y, W, lambd):
	temp = X.dot(W)
	# To avoid overflow
	for i in range(len(temp)):
		if temp[i]<-15:
			temp[i] = -15
			
	temp = 1 + np.exp(-temp)
	fx = 1/temp
	J = 0
	if (0 not in fx) and (1 not in fx):
		J = -(np.sum(Y.T.dot(np.log10(fx))) + np.sum((1-Y).T.dot(np.log10(1-fx)))) + lambd*np.sum(W ** 2)
	return J

# Function to do gradient descent logistic regression that returns the regression weights
def logistic_gradient_descent(X,Y,lambd):
	alpha = 0.0001
	# Stopping Criteria: Setting max_iterations incase it will stop converging to avoid infinite loop
	max_iterations = 10 ** 3
	# Stopping Criteria: Minimum cost difference after update to stop
	min_cost_diff = 0.0000001

	W = np.zeros(X.shape[1])
	n = len(Y)
	cost = cost_function(X,Y,W,lambd)
	for i in range(max_iterations):
		temp = X.dot(W)
		for j in range(len(temp)):
			if temp[j]<-15:
				temp[j] = -15

		temp = 1 + np.exp(-temp)
		Fx = 1/temp
		# Loss = hypothesis-actual
		loss = Fx-Y
		# Finding gradient of cost function wrt W
		grad = X.T.dot(loss) + 2*lambd*W
		# Updating W
		W = W - alpha*grad
		# Cost after update
		current_cost = cost_function(X,Y,W,lambd)
		if abs(current_cost-cost)<=min_cost_diff:
			print("No. of iterations: ",i)
			break
		if i==max_iterations-1:
			print("No. of iterations: 1000")

		cost = current_cost

	return W

# Function to do newton raphson logistic regression that returns the regression weights
def logistic_newton_raphson(X,Y,lambd):
	# Stopping Criteria: Setting max_iterations incase it will stop converging to avoid infinite loop
	max_iterations = 10 ** 3
	# Stopping Criteria: Minimum cost difference after update to stop
	min_cost_diff = 0.0000001

	W = np.zeros(X.shape[1])
	n = len(Y)
	cost = cost_function(X,Y,W,lambd)
	for i in range(max_iterations):
		temp = X.dot(W)
		# To avoid overflow
		for j in range(len(temp)):
			if temp[j]<-15:
				temp[j] = -15
		temp = 1 + np.exp(-temp)
		Fx = 1/temp
		
		# R for Hessian
		R = np.identity(n)
		for j in range(n):
			R[j][j] = Fx[j]*(1-Fx[j])

		# Updating W
		W = W - np.linalg.inv(X.T.dot(R.dot(X)) + 2*lambd*np.identity(len(W))).dot(X.T.dot(Fx - Y) + 2*lambd*W)
		# Cost after update
		current_cost = cost_function(X,Y,W,lambd)
		if abs(current_cost-cost)<=min_cost_diff:
			print("No. of iterations: ",i)
			break

		cost = current_cost

	return W

# Function that returns a prediction of the target variable given the input variables and regression weights.
def reg_output(X, weights):
	temp = X.dot(weights)
	# To avoid overflow
	for j in range(len(temp)):
		if temp[j]<-15:
			temp[j] = -15
	temp = 1 + np.exp(-temp)
	Fx = 1/temp
	return np.round(Fx)

# Function to check accuracy
def calc_accuracy(Y,Y_pred):
	return 100-np.mean(abs(Y-Y_pred))*100

def featuretransform(X, degree):
	n = len(X)
	if len(X.shape)>1:
		X1 = np.ones(n)
	else:
		X1 = np.ones(1)
	for i in range(degree+1):
		for j in range(degree+1):
			if i+j<=degree and i+j>0:
				if len(X.shape)>1:
					X1 = np.c_[X1 , (X[:,1] ** i)*(X[:,2] ** j)]
				else:
					X1 = np.c_[X1 , (X[1] ** i)*(X[2] ** j)]


	return X1


def plot_decision_boundary(data,X,degree,W,title):
	n = len(data)
	x1_yes = []
	x2_yes = []
	x1_no = []
	x2_no = []

	for i in range(n):
		if data[i][2]=='1':
			x1_yes.append(data[i][0])
			x2_yes.append(data[i][1])
		elif data[i][2]=='0':
			x1_no.append(data[i][0])
			x2_no.append(data[i][1])

	x1_no = np.array(x1_no).astype(np.float)
	x1_yes = np.array(x1_yes).astype(np.float)
	x2_yes = np.array(x2_yes).astype(np.float)
	x2_no = np.array(x2_no).astype(np.float)

	if X.shape[1]==3:
		# Only two points needed to draw a line
		x1 = [min(X[:,1])-0.5,max(X[:,1])+0.5]
		x1 = np.array(x1).astype(np.float)
		x2 = (-1/W[2])*(W[1]*x1+W[0])

	else:
		u = np.linspace(min(min(x1_yes),min(x1_no)),max(max(x1_no),max(x1_yes)),50)
		v = np.linspace(min(min(x2_no),min(x2_yes)),max(max(x2_no),max(x2_yes)),50)
		z = np.zeros([len(u),len(v)])

		for i in range(len(u)):
			for j in range(len(v)):
				z[i][j] = featuretransform(np.array([1,u[i],v[j]]).T,degree).dot(W)				

		z = z.T


	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(x1_no, x2_no, s=60, c='red',label='Rejected')
	ax1.scatter(x1_yes, x2_yes, s=60, c='blue',label='Accepted')
	if X.shape[1]==3:
		ax1.plot(x1,x2,c='blue')
	else:
		ax1.contour(u,v,z,0,linewidths=2)

	plt.xlim(min(min(x1_yes),min(x1_no)),max(max(x1_no),max(x1_yes)))
	plt.ylim(min(min(x2_no),min(x2_yes)),max(max(x2_no),max(x2_yes)))	
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title(title)
	plt.legend(loc='upper right')
	plt.show()


training_data,testing_data = data_partition(data,0.8)

X,Y = get_X_Y(training_data)
X_test,Y_test = get_X_Y(testing_data)
X = featuretransform(X,DEGREE)
X_test = featuretransform(X_test,DEGREE)

plot_data(training_data)

print("Degree = ",DEGREE,"\tLambda = ",LAMBD)
print("Newton Raphson:")
W = logistic_newton_raphson(X,Y,LAMBD)
Y_pred = reg_output(X,W)
print('Training Accuracy = ',calc_accuracy(Y,Y_pred))
Y_pred = reg_output(X_test,W)
print('Testing Accuracy = ',calc_accuracy(Y_test,Y_pred))
plot_decision_boundary(data,X,DEGREE,W,'Decision Boundary')

print("\nGradient Descent:")
W = logistic_gradient_descent(X,Y,LAMBD)
Y_pred = reg_output(X,W)
print('Training Accuracy = ',calc_accuracy(Y,Y_pred))
Y_pred = reg_output(X_test,W)
print('Testing Accuracy = ',calc_accuracy(Y_test,Y_pred))