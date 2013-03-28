#################################################
#Author: Michelle Austria Fernandez
#Uni: mf2492
#
#Program: nearestneighbor.py
#Overview: Contains functions that runs the
#NN simulations. KNN function for part B.
#################################################

import numpy as np
import matplotlib.pyplot as plt
import os


def readFile(file):
    '''Read in  data file'''
    f = open(file, 'r')
    line = f.readline() 
    malignant = []
    benign = []
    countm = 0
    countb = 0

    while line:
        array_line = line.rstrip('\n').split(',')
        array_line.pop(0)
        if (array_line[0] == 'M'):
            array_line.pop(0)
            malignant.append(array_line)
            countm = countm + 1
        elif (array_line[0] == 'B'):
            array_line.pop(0)
            benign.append(array_line)
            countb = countb + 1
        line = f.readline()
    malignant_array = np.vstack(malignant)
    benign_array = np.vstack(benign)

    f.close()
    return malignant_array, benign_array, countm, countb
    

    

def n_validator(test_data, p, classifier, *args):#,*args
    '''assesses the accuracy of the NNclassifier'''
    np.random.shuffle(test_data)
    score = 0
    partitions = test_data.shape[0] / p
    size = test_data.shape[1]

    test = np.array_split(test_data, p) #splits array into p equal parts
    test = np.asarray(test)

    for k in range(test.shape[0]): 
        p_tester = test[k]
        remaining = np.delete(test, k, axis=0)
        remaining = np.vstack(remaining)

        test_classifier= np.zeros(shape=(partitions, size)) # test data
        for i in range(test_classifier.shape[0]):
            test_classifier[i] = p_tester[i]
            
        training_classifier = np.zeros(shape=((partitions * (p-1)), size))
        for i in range(training_classifier.shape[0]):
            training_classifier[i] = remaining[i]

        labels = classifier(training_classifier, test_classifier, *args)
        original_labels = test_classifier[:,(size - 1)]
        
        for i in range(labels.shape[0]):
            if (labels[i] == original_labels[i]): #compare labels
                score = score + 1
    return score / float(test_data.shape[0])

    

def NNclassifier(training, test):
    '''find the nearest neighbor and returns class label'''
    size = training.shape[1]

    x = np.delete(training, size-1, 1)
    print x
    y = np.delete(test, size-1, 1)
    print y
    a = np.zeros(shape=((training.shape[0]),1))
    jlabel = np.zeros(shape=((test.shape[0])))
    print a

    for i in range(test.shape[0]):
        for h in range(training.shape[0]):
            distance = np.linalg.norm(x[h]-y[i]) #find Euclidean distance
            a[h] = distance

        nearest = np.sort(a, axis = 0)# sorts array

        for j in range(a.shape[0]):
            if (nearest[0] == a[j]):
                index = j
        label = training[index]
        label= label[(label.shape[0]) - 1]
        jlabel[i] = label

    print jlabel
    return jlabel


def makeData(n1, n2):
    '''creates synthetic data'''
    mean1 = [2,5.5] # mean for class 1
    cov1 = [[1,1],[1,2]] #covariance for class 1
    x1 = np.random.multivariate_normal(mean1, cov1, n1)
    
    mean2 = [.5,1] # mean for class 2
    cov2 = [[1,0],[0,1]] #mean for class 2
    x2 = np.random.multivariate_normal(mean2, cov2, n2)

    return x1, x2


def labelData(n1, n2, x1, x2):
    '''labels the data'''
    label1 = np.zeros((n1,1))
    class1 = np.hstack((x1,label1)) #add label for Class 1: '0'

    label2 = np.ones((n2,1))
    class2 = np.hstack((x2,label2)) #add label for Class 2: '1'
    
    data = np.concatenate((class1, class2), axis=0)
    
    return data

def KNNclassifier(training, test, k):
    '''find the K-nearest neighbor and returns class label'''
    size = training.shape[1]
    x = np.delete(training, size-1, 1)
    y = np.delete(test, size-1, 1)
    a = np.zeros(shape=((training.shape[0]),1))
    jlabel = np.zeros(shape=((test.shape[0])))

    for i in range(test.shape[0]):
        for h in range(training.shape[0]):
            distance = np.linalg.norm(x[h]-y[i]) #find Euclidean distance
            a[h] = distance

        nearest = np.sort(a, axis = 0)# sorts array

        class1 = 0
        class2 = 0

        for j in range(k):
            for m in range(a.shape[0]):
                if (nearest[j] == a[m]):
                    index = m
                    label = training[index]
                    label= label[(label.shape[0]) - 1]
                    if (label == 0):
                        class1 = class1 + 1
                    else:
                        class2 = class2 + 1
        if (class1 > class2):
            jlabel[i] = 0
        else:
            jlabel[i] = 1

    #print jlabel - commented out to save time when running the program.
    return jlabel




    
#####################################
#Function not used in this program
#Originally used for creating a visual 
#for the synthetic data.
#####################################
def makePlot(x1, x2, y1, y2):

    fig = plt.figure(1) # plot
    plot1 = plt.plot(x1[:, 0], x1[:, 1], 'ob', x2[:, 0], x2[:, 1], 'or')
    plot2 = plt.plot(y1[:, 0], y1[:, 1], 'og', y2[:, 0], y2[:, 1], 'oy')

    plt.show()
   
