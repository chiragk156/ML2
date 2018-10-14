Name: Chirag Khurana
Entry No.: 2016CSB1037
CSL603-Machine Learning
Lab-2
Python3

================================================================How to run my code===================================================
Libraries Required:

	0. Basic Libraries: math,random
	1. numpy
	To install in linux: $ sudo apt install python3-numpy
	2. matplotlib
	To install in linux: $ sudo apt install python3-matplotlib

To do preprocessing for "linregdata"(ALREADY DONE):
	
	$ python3 preprocessing.py
	
	Note: File "linregdata" must be present in same directory. Or change file path defined at top of the file
	It will partition the data into training and testing set.(80%-20%)


To run Linear Ridge Regression:
	
	$ python3 linear_regression.py

	Note: File "trainingdata.csv" and "testdata.csv" must be present in current directory or change the file path in variable names defined at top of the file.
	
	It will print Training and Testing MSE for the combinations defined at the top of the file. "frac_set" and "lambd_set" defined at the top.

	It will also plot graphs given in 7-9 steps of the assignment PDF.

	I have implemented functions with same name as given in the assignment. I have implemented both gradient descent and analytical methods, but used analytical method for 6-9 steps of the assignment to do faster calculation.
	

To run Regularized Logistic Regression:
	
	$ python3 logistic_regression.py

	To vary feature transform degree and lambda, change variable DEGREE and LAMBD defined at top of the file.

	Note: File "credit.txt" must be present in current directory change the file path in variable name defined at top of the file.

	It will give you the accuracy and also will plot the graphs. Data is partitioned into training and testing data(80%-20%).

======================================================================================================================================