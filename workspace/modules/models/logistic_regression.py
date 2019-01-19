import torch.nn as nn
from torch.utils.data.dataset import Dataset
import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression as sk_lr


def sigmoid(Z):
    return 1/(1+np.e**(-Z))


def logistic_loss(y, y_hat):
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))


def set_pairs(X):
    # input list of single x1 values and make 2d vectors with x2 =1
    X_2d = []
    for x1 in X:
        X_2d.append([x1, 1])
    return X_2d


class LogisticRegression:
    """
    One-to-One Network Model that trains logistic regression threshold
    to predict binary classification : Y=(0/1) for X=(x1)
    """

    def __init__(self, config=None):
        if config != None:
            self.epochs = config["epochs"]
            self.learning_rate = config["learning_rate"]
        else:
            self.epochs = 50
            self.learning_rate = 0.01
        self.T = 0 #treshold

    def train(self, X, Y):
        #not really training, just finign average threshold
        d_0 = [] # cln should have higher accuracies
        d_1 = [] # def
        for i in range(len(X)):
            if Y[i] == 0:
                d_0.append(X[i])
            elif Y[i] == 1:
                d_1.append(X[i])
            else:
                print("something wrong with training data")
        m_0 = np.average(d_0)
        m_1 = np.average(d_1)
        diff = m_0-m_1
        self.T = m_1+ diff*(len(d_1)/len(X))
        #print(self.T)
        

    def test(self, X):
        """
        predicts Y for X on trained model
        :param X: 2b-predicting float input
        :returns: true or false (for defect prediction true if defective)
        """
        if X>self.T:
            return 0#not buggy
        else:
            return 1
