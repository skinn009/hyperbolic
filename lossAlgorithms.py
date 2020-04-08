import numpy as np
from utilities import minkowskiDot, minkowskiArrayDot


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
        return np.matmul(self.examples.T, -(minkowskiArrayDot(self.examples, self.centroid) ** 2 - 1) ** -.5)

    def computeTangent(self):
        """
        Compute the gradient in the tangent space, Eq. (5).
        :return: ? np array k X 1 ?
        """
        return self.gradAmbient + self.centroid * minkowskiDot(self.centroid, self.gradAmbient)

    def computeLoss(self):
        """
        Sum of the distances between the points and the current centroid, in hyperbolic space.
        :return: scalar, which is the sum of the distances from the centroid to each of the points.
        """
        return sum(np.arccosh(-minkowskiArrayDot(self.examples, self.centroid))**2)[0]/np.shape(self.examples)[0]
