import numpy as np
from utilities import minkowskiDot, minkowskiArrayDot, generatePoints, randTheta, hyperboloidDist
import copy


class HyperGradLoss:
    """
    Grad descent in hyperbolic space w.r.t. the Frechet mean.
    """
    def __init__(self, X, theta):
        self.examples = X
        self.centroid = theta.reshape((X.shape[1], -1))
        self.loss = self.computeLoss()

        # compute grad in ambient
        self.gradAmbient = self.computeAmbient()
        # compute grad in tangent space
        self.gradTangent = self.computeTangent()

    def computeAmbient(self):
        """
        The gradient of the distance function in the ambient space, eq. (4), Wilson, Leimeister.
        :return: np array, (k + 1) x 1
        """
        mod = np.ones(self.centroid.shape)
        mod[-1] = -1
        array_dot = minkowskiArrayDot(self.examples, self.centroid)
        dMDP_dcent = (self.examples.T*mod).T.reshape(self.examples.shape)
        distances = np.arccosh(-array_dot)
        #return np.matmul(dMDP_dcent.T, -np.arccosh(-array_dot)*(array_dot ** 2 - 1) ** -.5)/self.examples.shape[1]
        #return np.matmul(self.examples.T, -(array_dot ** 2 - 1) ** -.5)/self.examples.shape[1]
        return -distances/np.sqrt(array_dot ** 2 - 1)


    def computeTangent(self):
        """
        Compute the gradient in the tangent space, Eq. (5).
        :return: np array k+1 X 1
        """
        return np.matmul(self.examples.T, self.gradAmbient)
        #return self.gradAmbient + self.centroid * minkowskiDot(self.centroid, self.gradAmbient)


    def computeLoss(self):
        """
        Avg of the distances^2 between the points and the current centroid, in hyperbolic space.
        :return: scalar, which is the avg of the distances^2 from the centroid to each of the points.
        """
        return sum(np.arccosh(-minkowskiArrayDot(self.examples, self.centroid))**2)[0]/np.shape(self.examples)[0]

if __name__ == "__main__":
    points = generatePoints(100)
    #print(points)
    theta = randTheta(2)
    #print("theta", theta.T)

    obj = HyperGradLoss(points, theta)
    print(obj.gradAmbient)
    print(obj.gradTangent)
    print(obj.loss)
    dist_list = []
    for point in points:
        dist_list.append(hyperboloidDist(point, theta)**2)
    print("distances^2 from centroid:\n", dist_list)
    print("avg", sum(dist_list)[0]/len(points))
