import random
import numpy as np
import math
import matplotlib.pyplot as plt


# The partition fraction set
frac_set = [0.03,0.06,0.1,0.2,0.5,0.8,1]
# lambda set
lambd_set = [0,0.1,0.5,1,2,4,8,16,20]


TRAINING_FILE_PATH = 'trainingdata.csv'
TESTING_FILE_PATH = 'testdata.csv'


# Partitioning the data into training and validation
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





# Separating X and Y from the training data and also returning mean and std of each xi. Amd Standardize the independent variables
def get_standard_X_Y(data):
	n = len(data)
	x0 = np.ones(n)
	x1 = []
	x2 = []
	x3 = []
	x4 = []
	x5 = []
	x6 = []
	x7 = []
	x8 = []
	x9 = []
	x10 = []
	Y = []
	for i in range(n):
		x1.append(data[i][0])
		x2.append(data[i][1])
		x3.append(data[i][2])
		x4.append(data[i][3])
		x5.append(data[i][4])
		x6.append(data[i][5])
		x7.append(data[i][6])
		x8.append(data[i][7])
		x9.append(data[i][8])
		x10.append(data[i][9])
		Y.append(data[i][10])

	x1 = np.array(x1).astype(np.float)
	x2 = np.array(x2).astype(np.float)
	x3 = np.array(x3).astype(np.float)
	x4 = np.array(x4).astype(np.float)
	x5 = np.array(x5).astype(np.float)
	x6 = np.array(x6).astype(np.float)
	x7 = np.array(x7).astype(np.float)
	x8 = np.array(x8).astype(np.float)
	x9 = np.array(x9).astype(np.float)
	x10 = np.array(x10).astype(np.float)

	
	# List to store means of each xi, i=1 to 10
	x_mean = []

	# List to store std of each xi, i=1 to 10
	x_std = []

	x1_mean = x1.mean()
	x_mean.append(x1_mean)
	x1_std = x1.std()
	x_std.append(x1_std)
	x1 = (x1-x1_mean)/x1_std

	x2_mean = x2.mean()
	x_mean.append(x2_mean)
	x2_std = x2.std()
	x_std.append(x2_std)
	x2 = (x2-x2_mean)/x2_std

	x3_mean = x3.mean()
	x_mean.append(x3_mean)
	x3_std = x3.std()
	x_std.append(x3_std)
	x3 = (x3-x3_mean)/x3_std

	x4_mean = x4.mean()
	x_mean.append(x4_mean)
	x4_std = x4.std()
	x_std.append(x4_std)
	x4 = (x4-x4_mean)/x4_std

	x5_mean = x5.mean()
	x_mean.append(x5_mean)
	x5_std = x5.std()
	x_std.append(x5_std)
	x5 = (x5-x5_mean)/x5_std

	x6_mean = x6.mean()
	x_mean.append(x6_mean)
	x6_std = x6.std()
	x_std.append(x6_std)
	x6 = (x6-x6_mean)/x6_std

	x7_mean = x7.mean()
	x_mean.append(x7_mean)
	x7_std = x7.std()
	x_std.append(x7_std)
	x7 = (x7-x7_mean)/x7_std

	x8_mean = x8.mean()
	x_mean.append(x8_mean)
	x8_std = x8.std()
	x_std.append(x8_std)
	x8 = (x8-x8_mean)/x8_std

	x9_mean = x9.mean()
	x_mean.append(x9_mean)
	x9_std = x9.std()
	x_std.append(x9_std)
	x9 = (x9-x9_mean)/x9_std

	x10_mean = x10.mean()
	x_mean.append(x10_mean)
	x10_std = x10.std()
	x_std.append(x10_std)
	x10 = (x10-x10_mean)/x10_std


	X = np.array([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]).T
	Y = np.array(Y).astype(np.float)
	
	return X,Y,x_mean,x_std



# Separating X and Y from the testing data and using mean and std of each training xi
def get_standard_test_X_Y(data,X_mean,X_std):
	n = len(data)
	x0 = np.ones(n)
	x1 = []
	x2 = []
	x3 = []
	x4 = []
	x5 = []
	x6 = []
	x7 = []
	x8 = []
	x9 = []
	x10 = []
	Y = []
	for i in range(n):
		x1.append(data[i][0])
		x2.append(data[i][1])
		x3.append(data[i][2])
		x4.append(data[i][3])
		x5.append(data[i][4])
		x6.append(data[i][5])
		x7.append(data[i][6])
		x8.append(data[i][7])
		x9.append(data[i][8])
		x10.append(data[i][9])
		Y.append(data[i][10])

	x1 = np.array(x1).astype(np.float)
	x2 = np.array(x2).astype(np.float)
	x3 = np.array(x3).astype(np.float)
	x4 = np.array(x4).astype(np.float)
	x5 = np.array(x5).astype(np.float)
	x6 = np.array(x6).astype(np.float)
	x7 = np.array(x7).astype(np.float)
	x8 = np.array(x8).astype(np.float)
	x9 = np.array(x9).astype(np.float)
	x10 = np.array(x10).astype(np.float)

	

	x1 = (x1-X_mean[0])/X_std[0]
	x2 = (x2-X_mean[1])/X_std[1]
	x3 = (x3-X_mean[2])/X_std[2]
	x4 = (x4-X_mean[3])/X_std[3]
	x5 = (x5-X_mean[4])/X_std[4]
	x6 = (x6-X_mean[5])/X_std[5]
	x7 = (x7-X_mean[6])/X_std[6]
	x8 = (x8-X_mean[7])/X_std[7]
	x9 = (x9-X_mean[8])/X_std[8]
	x10 = (x10-X_mean[9])/X_std[9]
	

	X = np.array([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]).T
	Y = np.array(Y).astype(np.float)
	
	return X,Y



# Function to calculate cost
def cost_function(X, Y, W, lambd):
    J = np.sum((X.dot(W) - Y) ** 2) + lambd*np.sum(W ** 2)
    return J


# Function named mylinridgereg(X, Y, lambda) that calculates the linear least squares solution with the ridge regression penalty parameter lambda (λ) and returns the regression weights
def mylinridgereg_gradient_descent(X, Y, lambd):
	# Gradient Descent
	alpha = 0.0001

	# Setting max_iterations incase it will stop converging to avoid infinite loop
	max_iterations = 10 ** 5

	# Minimum cost difference after update to stop
	min_cost_diff = 0.001

	W = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	n = len(Y)
	cost = cost_function(X,Y,W,lambd)

	for i in range(max_iterations):
		# Loss = hypothesis-actual
		loss = X.dot(W)-Y
		# Finding gradient of cost function wrt W
		grad = X.T.dot(loss)/n + 2*lambd*W/n
		# Updating W
		W = W - alpha*grad
		# Cost after update
		current_cost = cost_function(X,Y,W,lambd)
		if abs(current_cost-cost)<=min_cost_diff:
			break

		cost = current_cost

	return W


# Function named mylinridgereg(X, Y, lambda) that calculates the linear least squares solution with the ridge regression penalty parameter lambda (λ) and returns the regression weights
def mylinridgereg(X, Y, lambd):
	# Analytical Method

	W = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(Y))

	return W


# Function that returns a prediction of the target variable given the input variables and regression weights.
def mylinridgeregeval(X, weights):
	return X.dot(weights)

# Function that computes the mean squared error between the predicted and actual target values
def meansquarederr(T, Tdash):
	return np.mean((T-Tdash) ** 2)



# Training Data reading from file
file = open(TRAINING_FILE_PATH,'r')
data = file.read().split('\n')
file.close()
if '' in data:
	data.remove('')

n=len(data)

for i in range(n):
	data[i]=data[i].split(',')





# Testing Data reading from file
file = open(TESTING_FILE_PATH,'r')
test_data = file.read().split('\n')
file.close()
if '' in test_data:
	test_data.remove('')

n=len(test_data)

for i in range(n):
	test_data[i]=test_data[i].split(',')





# Experiment 6,7


training_mse_avg_for_each_frac = []
testing_mse_avg_for_each_frac = []

print("Linear Ridge Regression: #6\n")
for frac in frac_set:
	training_mse_avg = []
	testing_mse_avg = []
	for lambd in lambd_set:
		training_mse = []
		testing_mse = []
		for i in range(100):
			training_data,validation_data = data_partition(data,frac)

			X,Y,X_mean,X_std = get_standard_X_Y(training_data)
			W = mylinridgereg(X,Y,lambd)
			Y_pred = mylinridgeregeval(X,W)
			if meansquarederr(Y,Y_pred)>100:
				continue
			training_mse.append(meansquarederr(Y,Y_pred))

			X_test,Y_test = get_standard_test_X_Y(test_data,X_mean,X_std)
			Y_pred = mylinridgeregeval(X_test,W)
			testing_mse.append(meansquarederr(Y_test,Y_pred))
			
		training_mse = np.array(training_mse).astype(np.float)
		testing_mse = np.array(testing_mse).astype(np.float)
		print("frac = ",frac,"\tlambda = ",lambd)
		avg_training_mse = training_mse.mean()
		avg_testing_mse = testing_mse.mean()
		print("Training Avg. MSE = ",avg_training_mse)
		print("Testing  Avg. MSE = ",avg_testing_mse,"\n")
		training_mse_avg.append(avg_training_mse)
		testing_mse_avg.append(avg_testing_mse)

	training_mse_avg_for_each_frac.append(training_mse_avg)
	testing_mse_avg_for_each_frac.append(testing_mse_avg)


for i in range(len(frac_set)):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	plt.xlim(0,max(lambd_set))
	plt.ylim(0,max(max(max(testing_mse_avg_for_each_frac)),max(max(training_mse_avg_for_each_frac)))+2)
	ax1.plot(lambd_set,training_mse_avg_for_each_frac[i],'xb-',label='Training Avg. MSE')
	ax1.plot(lambd_set,testing_mse_avg_for_each_frac[i],'xr-',label='Testing Avg. MSE')
	plt.xlabel('Lambda')
	plt.ylabel('Avg. MSE')
	plt.title('#7 Training Fraction = '+str(frac_set[i])+", Instances = "+str(round(frac_set[i]*len(data))))
	plt.legend(loc='upper right')
	plt.show()



# Experiment 8
min_avg_mse_for_each_frac = []
lambd_for_min_mse = []
for x in testing_mse_avg_for_each_frac:
	min_avg_mse_for_each_frac.append(min(x))
	for i in range(len(x)):
		if x[i]==min(x):
			lambd_for_min_mse.append(lambd_set[i])
			break

fig = plt.figure()
plt.xlim(0,1.0)
plt.ylim(min(min_avg_mse_for_each_frac)-0.1,max(min_avg_mse_for_each_frac)+0.1)
plt.plot(frac_set,min_avg_mse_for_each_frac,'xb-')
plt.xlabel('Training Partition Fraction')
plt.ylabel('Minimum Avg. Testing MSE')
plt.title("#8")
plt.show()


fig = plt.figure()
plt.xlim(0,1.0)
plt.ylim(0,max(lambd_set))
plt.plot(frac_set,lambd_for_min_mse,'xb-')
plt.xlabel('Training Partition Fraction')
plt.ylabel('Lambda for Min. Avg. Testing MSE')
plt.title("#8")
plt.show()


# Experiment 9
# Training
training_min_mse_lambd = -1
training_min_mse_frac = 0
for i in range(len(training_mse_avg_for_each_frac)):
	for j in range(len(training_mse_avg_for_each_frac[i])):
		if training_mse_avg_for_each_frac[i][j]==min(min(training_mse_avg_for_each_frac)):
			training_min_mse_frac = frac_set[i]
			training_min_mse_lambd = lambd_set[j]
			break


training_data,validation_data = data_partition(data,training_min_mse_frac)

X,Y,X_mean,X_std = get_standard_X_Y(training_data)
W = mylinridgereg(X,Y,training_min_mse_lambd)
Y_pred = mylinridgeregeval(X,W)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(Y, Y_pred, s=5, c='red')
ax1.plot(range(math.ceil(max(max(Y_pred),max(Y)))),range(math.ceil(max(max(Y_pred),max(Y)))),c='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('#9 Training Data | lambda = '+str(training_min_mse_lambd)+' | frac = '+str(training_min_mse_frac))
plt.show()


# Testing
testing_min_mse_lambd = -1
testing_min_mse_frac = 0
for i in range(len(testing_mse_avg_for_each_frac)):
	for j in range(len(testing_mse_avg_for_each_frac[i])):
		if testing_mse_avg_for_each_frac[i][j]==min(min(testing_mse_avg_for_each_frac)):
			testing_min_mse_frac = frac_set[i]
			testing_min_mse_lambd = lambd_set[j]
			break


training_data,validation_data = data_partition(data,testing_min_mse_frac)

X,Y,X_mean,X_std = get_standard_X_Y(training_data)
W = mylinridgereg(X,Y,testing_min_mse_lambd)
X_test,Y_test = get_standard_test_X_Y(test_data,X_mean,X_std)
Y_pred = mylinridgeregeval(X_test,W)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(Y_test, Y_pred, s=5, c='red')
ax1.plot(range(math.ceil(max(max(Y_pred),max(Y_test)))),range(math.ceil(max(max(Y_pred),max(Y_test)))),c='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('#9 Testing Data | lambda = '+str(testing_min_mse_lambd)+' | frac = '+str(testing_min_mse_frac))
plt.show()