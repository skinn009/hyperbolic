import numpy as np
import math
import copy
import matplotlib.pyplot as plt


def generatePoints(N, k=2):
    """
    We generate N points on the hyperboloid satisfying the equation . If we are in H^2,
    We get random numbers for x1, x2, and compute for x3 where x3 = sqrt(1 + x1^2 + x2^2).
    Here, x3 represents the "height" of the point, in Minkowski space. So, the points are in H^2 on the hyperboloid.
    :param N:
    :return: numpy array of dimension N x k+1.
    """
    rands = [[np.random.rand() for j in range(k)] for i in range(N)]
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
    return sum([point1[i] * point2[i] for i in range(len(point1) - 1)]) - point1[-1] * point2[-1]


def minkowskiArrayDot(X, vec):
    """
    Computes the minkowski dot between N x K array and  vector. We multiply the last column of X by -1,
    then do normal matrix multiplication.
    :param array: N x k array
    :param vec: vector- reshaped to k X 1 for the matrix multiplication.
    :return: N x 1 array
    """
    k = X.shape[1]
    vec = vec.reshape((k, -1))
    X[:, -1] *= -1
    return np.matmul(X, vec)


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
        ax.plot(len(loss_val_array), loss_val_array, label=method_name)
    ax.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Grad Descent in Hyperbolic Space')
    plt.savefig(save_name)


if __name__ == "__main__":
    points = generatePoints(2)
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
