import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    
    f1 = X[i][0] * theta[0]
    f2 = sum([np.power(X[i][1], j) * theta[j] for j in range(1, len(theta))])
    hypothesis = f1 + f2
    ########################################/
    
    return hypothesis
