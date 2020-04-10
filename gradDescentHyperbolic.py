import numpy as np
import numpy.linalg as la

from utilities import hyperboloidDist, minkowskiDot, generatePoints, randTheta, plot_loss
from lossAlgorithms import HyperGradLoss


def hyperGradDescent(loss_object, theta, maxEvals, alpha, X, verbosity=True):
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
    grad_inf_norm = 1  # norm of the tangent grad
    prev_centroid = theta
    centroid_list.append(theta)
    while grad_inf_norm > 1e-5 and its <= maxEvals:
        curr_obj = loss_object(X, theta)
        # print("loss", curr_obj.loss)
        theta = exponentialMap(-alpha * curr_obj.gradTangent, curr_obj.centroid)
        # print("theta", theta)
        # diff_inf_norm = la.norm(prev_centroid - theta, np.inf)
        grad_inf_norm = la.norm(curr_obj.gradTangent, np.inf)
        loss_values.append(curr_obj.loss)
        centroid_list.append(theta)

        if verbosity == True and its% 10 == 0:
            print(its, curr_obj.loss, grad_inf_norm, curr_obj.centroid.T)
        its += 1
        prev_centroid = theta
    return loss_values, centroid_list


def armijoGradDescent (loss_object, theta, maxEvals, gamma, X, verbosity=True):
    """
    This is where we iteratively learn the centroid of the points. In this method, using the armijo methode to
    optimize the learning rate.
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
    grad_inf_norm = 1  # infinity norm of gradTangent, which we hope will approach 0
    prev_obj = loss_object(X, theta)
    centroid_list.append(theta)
    alpha = 1/la.norm(prev_obj.gradTangent)
    print("alpha", alpha)
    while grad_inf_norm > 1e-5 and its <= maxEvals:
        theta_p = exponentialMap(-alpha * prev_obj.gradTangent, prev_obj.centroid)
        curr_obj = loss_object(X, theta_p)
        while curr_obj.loss > prev_obj.loss - gamma * alpha * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent):
            alpha = alpha**2 * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent)\
                    / (2 * (curr_obj.loss + alpha * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent) - prev_obj.loss))
            print("alpha", alpha)
            print("gradient", curr_obj.gradTangent.T)
            print("loss", curr_obj.loss)
            theta_p = exponentialMap(-alpha * curr_obj.gradTangent, curr_obj.centroid)
            curr_obj = loss_object(X, theta_p)
            its +=1
            print(its, curr_obj.loss, curr_obj.gradTangent.T)

        grad_inf_norm = la.norm(curr_obj.gradTangent, np.inf)
        loss_values.append(curr_obj.loss)
        centroid_list.append(theta_p)

        if verbosity == True and its% 10 == 0:
            print(its, curr_obj.loss, grad_inf_norm, curr_obj.centroid.T)
        its += 1
        alpha = min(1, 2 * (prev_obj.loss - curr_obj.loss)/np.dot(curr_obj.gradTangent.T, curr_obj.gradTangent))
        prev_obj = curr_obj
    return loss_values, centroid_list


def exponentialMap(grad, p):
    """
    Compute the exponential map at p in H^n for some point grad. Exponential maps a point grad from the tangent space
    back onto the hyperboloid.

    :param grad: input to exponential map
    :param p: point to compute exponential map at
    :return: Point on H^k which corresponds to the exponential map at p evaluated at grad.
    """
    g_norm = la.norm(grad)
    return np.cosh(g_norm) * p + np.sinh(g_norm) * grad / g_norm


if __name__ == "__main__":
    points = generatePoints(3)
    print(points)
    # print(hyperboloidDist(points[0], points[1]))
    # theta = copy.deepcopy(points[0])
    theta = randTheta(2)


    loss_values, centroid_list = hyperGradDescent(HyperGradLoss, theta, 400, 0.2, points, True)
    print("initial loss", loss_values[0])
    

    #plot_loss({'grad descent': loss_values}, '/Users/michaelskinner/Desktop/vanilla.png')
    cent = centroid_list[-1]
    dist_list = []
    for point in points:
        dist_list.append(hyperboloidDist(point, cent)**2)
    print("num its:", len(centroid_list))
    print("last centroid:\n", cent)
    print("distances^2 from centroid:\n", dist_list)
    print("avg", sum(dist_list)/3)

    """
    
    loss_values, centroid_list = armijoGradDescent(HyperGradLoss, theta, 500, .1, points, True)
    """
