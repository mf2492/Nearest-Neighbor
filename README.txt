Michelle Fernandez
mf2492

This program contains two files: A6-B.py and nearestneighbor.py. In order to run this program, make sure that these two files are in the same folder directory since nearestneighbor.py is an imported class in A6-B.py.

nearestneighbor.py contains the two main functions for this program which are the k-nearest neighbor classifier and the validator, which tests the classifier's performance.

Running A6-B.py tests out two sets of data: one synthetic and one real set. The synthetic data uses a numpy function to create 500 different sets of data point, each one pertaining to a determined mean value. The real data comes from the Breast Cancer Wisconsin data set: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
The two different classes, malignant and benign, are labeled either 0 (malignant) or 1 (benign). 

To show the program in process, and in order to cut down run time, it only prints out the scores associated with each odd k-value between 1 and 15. And then it prints out the best K-value of each set.

The validator estimates the classifiers' performance as follows:
Synthetic data: ~ 97.8% (K=15)
Real data: ~ 91.9% (K=11)

