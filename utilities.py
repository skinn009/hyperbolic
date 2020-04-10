import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import random


def generatePoints(N, k=2, scale=1):
    """
    We generate N points on the hyperboloid satisfying the equation . If we are in H^2,
    We get random numbers for x1, x2, and compute for x3 where x3 = sqrt(1 + x1^2 + x2^2).
    Here, x3 represents the "height" of the point, in Minkowski space. So, the points are in H^2 on the hyperboloid.
    :param N:
    :return: numpy array of dimension N x k+1.
    """
    random.seed(30)
    rands = [[np.random.uniform(-scale, scale) for j in range(k)] for i in range(N)]
    point_list = []
    for rand in rands:
        # lastItem = math.sqrt(sum([1 + item**2 for item in rand]))
        lastItem = math.sqrt(1 + np.dot(rand, rand))
        rand.append(lastItem)
        point_list.append(rand)
    return np.array(point_list)


def randTheta(k):
    """
    Generates a random point on the hyperboloid
    :param k: 2 for these experiments
    :return: k + 1 X 1 array
    """
    return generatePoints(1, k)[0].reshape((k + 1, -1))


def minkowskiDot(point1, point2):
    point1 = list(point1)
    point2 = list(point2)
    mdp = sum([point1[i] * point2[i] for i in range(len(point1) - 1)]) - point1[-1] * point2[-1]
    return min(mdp, -1e-10)


def minkowskiArrayDot(X, vec):
    """
    Computes the minkowski dot between N x K array and  vector. We multiply the last element of vec by -1,
    then do normal matrix multiplication. To prevent errors, we set a ceiling on MDP to be -1e-10.
    :param array: N x k array
    :param vec: vector- reshaped to k X 1 for the matrix multiplication.
    :return: N x 1 array
    """
    max_MDP = -1e-10
    k = X.shape[1]
    vec = vec.reshape((k, -1))
    mod = np.ones(vec.shape)
    mod[-1] = -1
    MDP_array = np.matmul(X, vec*mod)
    MDP_array[MDP_array > max_MDP] = max_MDP
    return MDP_array


def hyperboloidDist(point1, point2):
    """
    From the equation (2) in paper by Wilson, Leimeister. Points in H^k space.
    :param point1:
    :param point2:
    :return: Distance on the hyperboloid between the points.
    """
    return np.arccosh(-minkowskiDot(point1, point2))


def plot_loss(loss_values_dict, save_name):
    """
    Plot loss_values vs. iteration number.
    :param loss_values_dict: dictionary of loss value arrays keyed by name (i.e. 'grad_descent', 'armijo',...). Note
    that all loss value arrays must be of the same length.
    :param save_name: location to save image
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Add each method to the plot
    for (method_name, loss_val_array) in loss_values_dict.items():
        print(method_name, len(loss_val_array))
        ax.plot(range(len(loss_val_array)), loss_val_array, label=method_name)
    ax.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Grad Descent in Hyperbolic Space')
    plt.savefig(save_name)


def poincareDist(p1, p2):
    """
    From equation (6) in Wilson, Leimeister. Points in B^k space.
    :param p1: point in poincare space (np.array)
    :param p2: point in poincare space (np.array)
    :return: Distance between two points in poincare ball model space
    """
    return np.arccosh(1 + 2 * ((np.linalg.norm(p1 - p2) ** 2)
                               / (1 - (np.linalg.norm(p1) ** 2))
                               * (1 - (np.linalg.norm(p2) ** 2))))


def hyperbolic_to_poincare(x):
    """
    Maps a point in hyperbolic space to poincare space. Equation (7) Wilson, Leimeister.
    :param x: point in (n+1)-dim hyperbolic space
    :return: point in n-dim poincare space
    """
    return (1 / (x[-1] + 1)) * x[:-1]


def poincare_to_hyperbolic(y):
    """
    Maps a point in poincare space to hyperbolic space. Equation (8) Wilson, Leimeister.
    :param y: point in n-dim poincare space
    :return: point in (n+1)-dim hyperbolic space
    """
    norm = np.linalg.norm(y)
    x = np.zeros(y.shape)
    x[:-1] = y
    x[-1] = (1 + (norm ** 2)) / 2
    return (2 / (1 - (norm ** 2))) * x


def plot_poincare(points, save_name):
    """
    Plot given (poincare) points on a 2D grid, save_fig to save_name.
    :param points: np.array of size (N, k), with N points, each with k=2.
    :param save_name: directory to save figure
    :return: None
    """
    fig = plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    circle = plt.Circle((0, 0), 1., color='black', fill=False)
    ax = plt.gca()
    ax.add_artist(circle)
    plt.savefig(save_name)
    return


if __name__ == "__main__":
    random.seed(1)
    points = generatePoints(1)
    # print(points[0])
    b = copy.deepcopy(points[0])
    # b[:,-1] *= -1
    # print(b)
    # print(np.dot(points[0], points[0]))
    # print(-minkowskiDot(points[0], points[0]))
    # print(hyperboloidDist(points[0], points[0]))
    newpoint = b.reshape((points[0].shape[0], -1))
    # print(newpoint)
    print(newpoint.T[0])
    print(np.arccosh(-minkowskiArrayDot(newpoint.T, points[0])))
    print(randTheta(2))

    # Testing poincare plotting
    print(hyperbolic_to_poincare(np.array([1, 2, 3, 4, -5])))
    points = generatePoints(200, scale=15)
    poincare_points = np.array([hyperbolic_to_poincare(points[idx]) for idx in range(len(points))])
    print(poincare_points.shape)
    plot_poincare(poincare_points, 'plots/poincare_sample.png')

    # Three points in poincare space. B and C should be same distance away from the origin.
    # Furthermore, B and C should be further apart from each other than they are from the origin. i.e. the geodesic
    # from B->C should not be the straight path, it must arc towards the origin.
    A = np.array([0., 0.])
    B = np.array([0., 0.5])
    C = np.array([-0.5, 0.])
    print(poincareDist(A, B), poincareDist(A, C), poincareDist(B, C))
