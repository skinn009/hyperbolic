
import numpy as np
import math

def generatePoints (N, k = 2):
    """
    We generate N points on the hyperboloid satisfying the equation . If we are in H^2,
    We get random numbers for x1, x2, and compute for x3 where x3 = sqrt(1 = x1^2 + x2^2).
    Here, x3 represents the "height" of the point, in Minkowski space. So, the points are in H^2 on the hyperboloid.
    :param N:
    :return: numpy array of dimension N x k+1.
    """
    rands = [[np.random.rand() for j in range(k)] for i in range(N)]
    point_list = []
    for rand in rands:
        lastItem = math.sqrt(sum([1 + item**2 for item in rand]))
        rand.append(lastItem)
        point_list.append(rand)
    return np.array(point_list)

def minkowskiDot(point1, point2):
    point1 = list(point1)
    point2 = list(point2)
    return sum([point1[i] * point2[i] for i in range(len(point1)-1)]) - point1[-1] * point2[-1]

def minkowskiArrayDot(X, vec):
    """
    Computes the minkowski dot between N x K array and 1 x k vector. We multiply the last column of X by -1,
    then do normal matrix multiplication.
    :param array:
    :param vec:
    :return:
    """
    X[:,-1] *= -1
    return np.matmul(X,vec)


def hyperboloidDist(point1, point2):
    """
    From the equation (2) in paper by Wilson, Leimeister. Points in H^k space.
    :param point1:
    :param point2:
    :return: Distance on the hyperboloid between the points.
    """
    return math.acosh(-minkowskiDot(point1,point2))

if __name__ == "__main__":
    points = generatePoints(2)
    print(hyperboloidDist(points[0], points[1]))

