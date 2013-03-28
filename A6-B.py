#################################################
#Author: Michelle Austria Fernandez
#Uni: mf2492
#
#Program: A6-B.py
#Overview: This program intakes both synthetic
#and real data and labels each data by a class
#category based off of a set of a training data
#which helps determine the nearest neighbor.
#The validator estimates the classifier's
#performance.
#################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import nearestneighbor as nn

def main ():

    p = 10 
    testsample1 = 250
    testsample2 = 250
    x1, x2 = nn.makeData(testsample1, testsample2)
    test_data = nn.labelData(testsample1, testsample2, x1, x2)
    print "Synthetic Data:"
    testK(test_data, p)

    m, b, sizem, sizeb = nn.readFile('wdbc.txt')
    test_data = nn.labelData(sizem, sizeb, m, b)
    print " "
    print "Real Data: "
    testK(test_data, p)


def testK(data, p):
    '''tests for K values 1-15'''
    bestValue = 0
    bestK = 0
    for k in range(1, 17, 2):
        print "K = ", k ,
        score = nn.n_validator(data, p, nn.KNNclassifier, k)
        print "SCORE: ", score
        if (score >= bestValue):
            bestK = k
            bestValue = score
        k = k + 2
    print "Best Value: "
    print "K = ", bestK
    print bestValue

   
main()
