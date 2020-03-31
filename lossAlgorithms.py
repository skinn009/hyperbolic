import numpy as np
from utilities import minkowskiDot
from utilities import minkowskiArrayDot


class hyperGradLoss:

    def __init__ (self, X, theta)
    self.examples = X
    self.centroid = theta
    self.loss = computeLoss()
    self.gradAmbient = computeAmbient()
    self.gradTangent = computeTangent()

    def computeAmbient(self):
        ambGrad = np.matmul(self.examples.T,
                         (minkowskiArrayDot(self.examples,self.centroid)**2 - 1)**-.5)
        return amdGrad

    def computeTangent(self):



    def computeLoss(self):
        """
        Sum of the distances between the points and the current centroid, in hyperbolic space.
        :return:
        """
        return sum(-np.arccosh(minkowskiArrayDot(self.examples, self.centroid)))


