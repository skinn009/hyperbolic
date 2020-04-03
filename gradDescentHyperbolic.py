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
    centroid_list = []
    grad_inf_norm = 1 # norm of difference between current centroid and previous centroid
    prev_centroid = theta
    centroid_list.append(theta)
    while grad_inf_norm > 1e-5 and its <= maxEvals:
        curr_obj = loss_object(X, theta)
        #print("loss", curr_obj.loss)
        theta = exponentialMap(-alpha*curr_obj.gradTangent, curr_obj.centroid)
        #print("theta", theta)
        #diff_inf_norm = la.norm(prev_centroid - theta, np.inf)
        grad_inf_norm = la.norm(curr_obj.gradTangent, np.inf)
        loss_values.append(curr_obj.loss)
        centroid_list.append(theta)
        if verbosity == True and its% 10 == 0:
            print(its, curr_obj.loss, grad_inf_norm, curr_obj.centroid.T)
        #print(its, curr_obj.loss, curr_obj.centroid.T)
        its += 1
        prev_centroid = theta
    return loss_values, centroid_list


def exponentialMap(grad, p):
    g_norm = la.norm(grad)
    return np.cosh(g_norm)*p + np.sinh(g_norm) * grad/g_norm





if __name__ == "__main__":
    points = generatePoints(5)
    print(points)
    #print(hyperboloidDist(points[0], points[1]))
    #theta = copy.deepcopy(points[0])
    theta = randTheta(2)
    #obj = hyperGradLoss(points, theta)
    """
    #To compute mean interations till convergence.
    conversion_its = []
    for i in range(500):
        points = generatePoints(10)
        loss_list, centroid_list = hyperGradDescent(hyperGradLoss, theta, 250, 0.01, points, True)
        conversion_its.append(len(centroid_list))
    print(sum(conversion_its)/len(conversion_its))
    """
    _, centroid_list = hyperGradDescent(hyperGradLoss, theta, 1000, 0.01, points, True)
    cent = centroid_list[-1]
    dist_list = []
    for point in points:
        dist_list.append(hyperboloidDist(point, cent))
    print("num its:", len(centroid_list))
    print("last centroid:\n", cent)
    print("distances from centroid:\n", dist_list)
    """
    #print(loss_list)
    #print(obj.centroid)
    #print(obj.loss)
    #print(obj.gradAmbient)
    """
