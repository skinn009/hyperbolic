import numpy as np
from utilities import hyperboloidDist
from utilities import minkowskiDot
from utilities import generatePoints
from utilities import randTheta
import copy

from lossAlgorithms import hyperGradLoss


def hyperGradDescent(loss_object, theta, maxEvals , alpha, X, verbosity = True):
    """
    This is where we iteratively learn the centroid of the points. In this method, we will stick with constant alpha.
    :param loss_object: this is the name of the loss function in lossAlgorithms.py, where the gradient and loss is
    computed at each iteration of the algorithm.
    :param theta: initial centroid
    :param maxEvals:
    :param alpha: learning rate
    :param X: N x k+1 array of the points on the hyperboloid. n + 1 (rather than n) because we are in Minkowski ambient.
    :param verbosity:
    :return:
    """
    return








if __name__ == "__main__":
    points = generatePoints(2)
    print(points)
    print(hyperboloidDist(points[0], points[1]))
    #theta = copy.deepcopy(points[0])
    theta = randTheta(2)
    obj = hyperGradLoss(points, theta)
    print(obj.centroid)
    print(obj.loss)
    print(obj.gradAmbient)
