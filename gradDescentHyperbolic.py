import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities import *
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
    centroid_list.append(theta)
    grad_inf_norm = 1  # norm of the tangent grad
    prev_centroid = theta
    # centroid_list.append(theta)
    while grad_inf_norm > 1e-4 and its <= maxEvals:
        curr_obj = loss_object(X, theta)
        # print("loss", curr_obj.loss)
        theta = exponentialMap(-alpha * curr_obj.gradTangent, curr_obj.centroid)
        # print("theta", theta)
        # grad_inf_norm = la.norm(prev_centroid - theta, np.inf)
        # grad_inf_norm = la.norm(curr_obj.gradTangent)
        grad_inf_norm = np.sqrt(minkowskiDot(curr_obj.gradTangent, curr_obj.gradTangent))
        loss_values.append(curr_obj.loss)
        centroid_list.append(theta)

        if verbosity == True and its % 10 == 0:
            print(its, curr_obj.loss, grad_inf_norm, curr_obj.centroid.T)
        its += 1
        prev_centroid = theta
    return loss_values, centroid_list


def armijoGradDescent(loss_object, theta, maxEvals, gamma, X, verbosity=True):
    """
    This is where we iteratively learn the centroid of the points. In this method, using the armijo method to
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
    alpha = 1 / la.norm(prev_obj.gradTangent)

    while grad_inf_norm > 1e-4 and its <= maxEvals:
        theta_p = exponentialMap(-alpha * prev_obj.gradTangent, prev_obj.centroid)
        curr_obj = loss_object(X, theta_p)
        while curr_obj.loss > prev_obj.loss - gamma * alpha * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent):
            alpha = alpha ** 2 * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent) \
                    / (2 * (curr_obj.loss + alpha * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent) - prev_obj.loss))
            # print("alpha", alpha)
            # print("gradient", curr_obj.gradTangent.T)
            # print("loss", curr_obj.loss)
            theta_p = exponentialMap(-alpha * curr_obj.gradTangent, curr_obj.centroid)
            curr_obj = loss_object(X, theta_p)
            its += 1
            print(its, curr_obj.loss, curr_obj.gradTangent.T)

        grad_inf_norm = la.norm(curr_obj.gradTangent, np.inf)
        loss_values.append(curr_obj.loss)
        centroid_list.append(theta_p)

        if verbosity == True and its % 10 == 0:
            print(its, curr_obj.loss, grad_inf_norm, curr_obj.centroid.T)
        its += 1
        alpha = min(1, 2 * (prev_obj.loss - curr_obj.loss) / np.dot(curr_obj.gradTangent.T, curr_obj.gradTangent))
        prev_obj = curr_obj
    return loss_values, centroid_list


def barzeliaBowrein(loss_object, theta, maxEvals, gamma, X, verbosity=True):
    its = 1
    loss_values = []
    centroid_list = []
    centroid_list.append(theta)
    grad_norm = 1
    prev_obj = loss_object(X, theta)
    # Initialize alpha to 1/||grad||
    alpha = 1 / la.norm(prev_obj.gradTangent)
    theta_prev = theta
    theta_cur = exponentialMap(-alpha * prev_obj.gradTangent, prev_obj.centroid)
    while its <= maxEvals and grad_norm > 1e-4:
        curr_obj = loss_object(X, theta_cur)
        while curr_obj.loss > prev_obj.loss - gamma * alpha * np.dot(prev_obj.gradTangent.T,
                                                                     prev_obj.gradTangent) and its <= maxEvals:
            alpha = np.dot((theta_cur - theta_prev).T, (theta_cur - theta_prev)) / np.dot(
                (prev_obj.gradTangent - curr_obj.gradTangent).T,
                (prev_obj.gradTangent - curr_obj.gradTangent))
            theta_p = exponentialMap(-alpha * prev_obj.gradTangent, prev_obj.centroid)
            curr_obj = loss_object(X, theta_p)
            # print(its, curr_obj.loss)
            its += 1
        theta = exponentialMap(-alpha * curr_obj.gradTangent, curr_obj.centroid)
        grad_norm = la.norm(curr_obj.gradTangent, np.inf)
        centroid_list.append(theta)
        if verbosity == True and its % 10 == 0:
            print(its, alpha, curr_obj.loss, grad_norm)
        its += 1
        theta_prev = theta_cur
        theta_cur = theta
        prev_obj = curr_obj
        loss_values.append(curr_obj.loss)
    # while len(loss_values) < maxEvals:
    #     loss_values.append(loss_values[-1])
    return loss_values, centroid_list


def nesterovAccGD(loss_object, theta, maxEvals, gamma, X, verbosity=True):
    its = 1
    loss_values = []
    centroid_list = []
    centroid_list.append(theta)
    grad_norm = 1
    prev_obj = loss_object(X, theta)
    lam_prev = 0
    alpha = 1 / la.norm(prev_obj.gradTangent)
    y_prev = exponentialMap(-alpha * prev_obj.gradTangent, prev_obj.centroid)
    while its <= maxEvals and grad_norm > 1e-4:
        curr_obj = loss_object(X, y_prev)
        # print(its, curr_obj.loss)
        lam_cur = (1 + math.sqrt(1 + 4 * lam_prev ** 2)) / 2
        # print(lam_cur, lam_prev)
        beta = (1 - lam_prev) / lam_cur
        while curr_obj.loss > prev_obj.loss - gamma * alpha * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent):
            alpha = alpha ** 2 * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent) \
                    / (2 * (
                    curr_obj.loss + alpha * np.dot(prev_obj.gradTangent.T, prev_obj.gradTangent) - prev_obj.loss))
            theta_p = exponentialMap(-alpha * curr_obj.gradTangent, curr_obj.centroid)
            curr_obj = loss_object(X, theta_p)
            its += 1
            # print("alpha", alpha)
        y_cur = exponentialMap(-alpha * curr_obj.gradTangent, curr_obj.centroid)
        # print(y_cur)
        theta = (1 - beta) * y_cur + beta * y_prev
        grad_norm = la.norm(curr_obj.gradTangent, np.inf)
        if verbosity >= 1 and its % 10 == 0:
            print(its, curr_obj.loss, grad_norm, curr_obj.centroid.T)
        its += 1
        lam_prev = lam_cur
        y_prev = y_cur
        prev_obj = curr_obj
        loss_values.append(curr_obj.loss)
        centroid_list.append(theta)

    return loss_values, centroid_list


def exponentialMap(grad, p):
    """
    Compute the exponential map at p in H^n for some point grad. Exponential maps a point grad from the tangent space
    back onto the hyperboloid.

    :param grad: input to exponential map
    :param p: point to compute exponential map at
    :return: Point on H^k which corresponds to the exponential map at p evaluated at grad.
    """
    # g_norm = la.norm(grad)
    # print ("mink_dot", minkowskiDot(grad, grad))
    g_norm = np.sqrt(max(minkowskiDot(grad, grad), 0))
    if g_norm == 0:
        return p

    # print("grad norm", g_norm)
    return (np.cosh(g_norm) * p) + (np.sinh(g_norm) / g_norm) * grad


def test_recursion():
    """
    Generate 2 pairs of points, compute the centroid for each pair, then generate a centroid from the pair of
    centroids, and determine the distances from the final centroid to all 4 points.
    :return:
    """
    points1 = generatePoints(2)
    theta1 = randTheta(2)
    loss_values1, centroid_list1 = hyperGradDescent(HyperGradLoss, theta1, 100, 0.4, points1, True)
    cent1 = centroid_list1[-1]

    points2 = generatePoints(2)
    theta2 = randTheta(2)
    loss_values2, centroid_list2 = hyperGradDescent(HyperGradLoss, theta2, 100, 0.4, points2, True)
    cent2 = centroid_list2[-1]

    points3 = np.array([cent1, cent2]).reshape(2, -1)
    theta3 = randTheta(2)
    loss_values3, centroid_list3 = hyperGradDescent(HyperGradLoss, theta3, 100, 0.4, points3, True)
    cent_final = centroid_list3[-1]

    points = np.concatenate((points1, points2), axis=0)
    print("points", points)
    dist_list = []
    for point in points:
        dist_list.append(hyperboloidDist(point, cent_final) ** 2)

    print("distance list:")
    print(dist_list)


def experimentOne():
    """
    Experiment 1 from our project writeup. Vanilla GD with 5 points in H^2.
    """
    points = generatePoints(50, k=2, scale=10)
    theta = randTheta(2, scale=50)
    loss_values, centroid_list = hyperGradDescent(HyperGradLoss, theta, 100, 0.1, points, True)
    plot_poincare(points, centroid_list, save_name='plots/experiment1_results.png')
    # plot_loss({'grad descent': loss_values}, 'plots/experiment1_loss.png')

    # We also plot the points and centroid in R^3 to show that we are not solving in euclidean space.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='g')
    ax.scatter(centroid_list[-1][0], centroid_list[-1][1], centroid_list[-1][2], color='red')
    plt.savefig('plots/experiment1_example.png')


def experimentTwo():
    """
    Experiment 2. We check if the
    """
    points = generatePoints(5, k=2, scale=10, same_quadrant=True)
    theta = np.reshape(points[0, :], (3, 1))
    loss_values, centroid_list = hyperGradDescent(HyperGradLoss, theta, 100, 0.1, points, True)
    plot_poincare(points, centroid_list, save_name='plots/experiment2_results.png')

    # We also plot the points and centroid in R^3 to show that we are not solving in euclidean space.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='g')
    ax.scatter(centroid_list[-1][0], centroid_list[-1][1], centroid_list[-1][2], color='red')
    plt.savefig('plots/experiment2_example.png')


def experimentThree():
    m = 200
    d = 2
    points = generatePoints(m, k=d, scale=1)
    # theta = np.reshape(points[0, :], (d + 1, 1))
    theta = randTheta(d, scale=1)
    loss_values_vanilla, cv = hyperGradDescent(HyperGradLoss, theta.copy(), 500, 0.1, points, True)
    loss_values_armijo, ca = armijoGradDescent(HyperGradLoss, theta.copy(), 500, 0.1, points, True)
    loss_values_bb, cb = barzeliaBowrein(HyperGradLoss, theta.copy(), 500, 0.1, points, True)
    loss_values_agd, cagd = nesterovAccGD(HyperGradLoss, theta.copy(), 500, 0.1, points, True)
    plot_loss({'Barzelia-Bowrein': loss_values_bb,
               'Vanilla': loss_values_vanilla,
               'Armijo Vanilla': loss_values_armijo,
               'Nesterov AGD': loss_values_agd},
              'plots/experiment_3_loss_comparison.png')

    plot_poincare(points, centroid_list=cv, save_name='plots/experiment_3_vanilla.png')
    plot_poincare(points, centroid_list=ca, save_name='plots/experiment_3_armijo.png')
    plot_poincare(points, centroid_list=cb, save_name='plots/experiment_3_bb.png')
    plot_poincare(points, centroid_list=cagd, save_name='plots/experiment_3_agd.png')

    print(cv[-1])
    print(ca[-1])
    print(cb[-1])
    print(cagd[-1])

def testing():
    points = generatePoints(200)
    # print(points)
    # print(hyperboloidDist(points[0], points[1]))
    theta = randTheta(2)

    loss_values, centroid_list = hyperGradDescent(HyperGradLoss, theta, 100, 0.1, points, True)

    print("initial loss", loss_values[0])
    plot_poincare(points, centroid_list, save_name='plots/poincare_sample.png')
    plot_loss({'grad descent': loss_values}, 'plots/vanilla.png')

    cent = centroid_list[-1]
    dist_list = []
    # for point in points:
    #     dist_list.append(hyperboloidDist(point, cent) ** 2)
    # print("num its:", len(centroid_list))
    # print("last centroid:\n", cent)
    # print("distances^2 from centroid:\n", dist_list)
    # print("avg", sum(dist_list) / len(dist_list))

    loss_values, centroid_list = armijoGradDescent(HyperGradLoss, theta, 50, .1, points, True)
    print(centroid_list[-1])
    plot_poincare(points, centroid_list, save_name='plots/armijo_sample.png')
    plot_loss({'Armijo grad descent': loss_values}, 'plots/vanilla_armijo.png')

    loss_values, centroid_list = barzeliaBowrein(HyperGradLoss, theta, 200, 0.1, points, True)
    plot_poincare(points, centroid_list, save_name='plots/barzelia_sample.png')
    plot_loss({'barzeliaBowrein- grad descent': loss_values}, 'plots/vanilla_barzelia.png')

    loss_values, centroid_list = nesterovAccGD(HyperGradLoss, theta, 100, .1, points, True)
    plot_poincare(points, centroid_list, save_name='plots/nesterov.png')
    plot_loss({'Nesterov accelerated- grad descent': loss_values}, 'plots/vanilla_nestrov.png')


if __name__ == '__main__':
    # experimentOne()
    # experimentTwo()
    experimentThree()
    # testing()
