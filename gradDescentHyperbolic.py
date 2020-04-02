import numpy as np
import copy
import numpy.linalg as la

from utilities import hyperboloidDist
from utilities import minkowskiDot
from utilities import generatePoints
from utilities import randTheta
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
    its = 1
    loss_values = []
    diff_inf_norm = 1 # norm of difference between current centroid and previous centroid
    prev_centroid = theta
    while diff_inf_norm > 1e-4 and its <= maxEvals:
        curr_obj = loss_object(X, theta)
        #print("loss", curr_obj.loss)
        theta = exponentialMap(-alpha*curr_obj.gradTangent, curr_obj.centroid)
        #print("theta", theta)
        diff_inf_norm = la.norm(prev_centroid - theta, np.inf)
        loss_values.append(curr_obj.loss)
        if verbosity == True:# and its% 10 == 0:
            print(its, curr_obj.loss, diff_inf_norm, curr_obj.centroid.T)
        its += 1
        prev_centroid = theta
    return loss_values


def exponentialMap(grad,p):
    g_norm = la.norm(grad)
    return np.cosh(g_norm)*p + np.sinh(g_norm) * grad/g_norm





if __name__ == "__main__":
    points = generatePoints(20)
    print(points)
    #print(hyperboloidDist(points[0], points[1]))
    #theta = copy.deepcopy(points[0])
    theta = randTheta(2)
    #obj = hyperGradLoss(points, theta)
    loss_list = hyperGradDescent(hyperGradLoss, theta, 30, 0.01, points, True)
    #print(loss_list)
    #print(obj.centroid)
    #print(obj.loss)
    #print(obj.gradAmbient)
