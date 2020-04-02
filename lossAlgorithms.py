import numpy as np
from utilities import minkowskiDot
from utilities import minkowskiArrayDot
from utilities import exponentialMap


class hyperGradLoss:

    """
    Grad descent in hyperbolic space w.r.t. the Frechet mean.
    """

    def __init__(self, X, theta):

        self.examples = X
        self.centroid = theta.reshape((X.shape[1], -1))
        self.loss = self.computeLoss()

        # ---| One iteration of grad descent has the following steps (may move to gradDescentHyperbolic.py)
        # compute grad in ambient
        self.gradAmbient = self.computeAmbient()
        # compute grad in tangent space
        self.gradTangent = self.computeTangent()
        # now do gradient update on the hyperboloid
        #self.centroid = self.gradientUpdate()

    def computeAmbient(self):
        """
        The gradient of the distance function in the ambient space, eq. (4), Wilson, Leimeister.
        :return: np array, (k + 1) x 1
        """
        ambGrad = np.matmul(self.examples.T,

                            -(minkowskiArrayDot(self.examples, self.centroid)**2 - 1)**-.5)

        return ambGrad

    def computeTangent(self):
        """
        Compute the gradient in the tangent space, Eq. (5).
        :return: ? np array k X 1 ?
        """

        tangentGrad = self.gradAmbient + self.centroid * minkowskiDot(self.centroid, self.gradAmbient)

        return tangentGrad

    def gradientUpdate(self):
        """
        Perform the gradient update of param theta: theta = Exp_theta (-alpha * self.gradTangent)

        :return: new theta centroid parameter
        """
        newTheta = exponentialMap(self.centroid, self.gradTangent)
        return newTheta


    def computeLoss(self):
        """
        Sum of the distances between the points and the current centroid, in hyperbolic space.
        :return: scalar, which is the sum of the distances from the centroid to each of the points.
        """
        return sum(np.arccosh(-minkowskiArrayDot(self.examples, self.centroid)))[0]
