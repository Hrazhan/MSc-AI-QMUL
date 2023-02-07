import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    hypothesis = 0.0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    
    hypothesis = (theta[0] * X[i][0]) + (theta[1] * X[i][1])
    #########################################
    # print("I", i)
    # print("Hypothesis", hypothesis)
    # print("Theta", theta.shape)
    # print("X",X[i][0])
    # print(theta[0][0])    
    return hypothesis
