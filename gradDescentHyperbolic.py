import numpy as np
from utilities import hyperboloidDist
from utilities import minkowskiDot
from utilities import generatePoints

from lossAlgorithms import hyperGradLoss


def hyperGradDescent(loss_object, theta, maxEvals = 25, alpha, X, verbosity = True):








if __name__ == "__main__":
    points = generatePoints(2)
    print(points)
    print(hyperboloidDist(points[0], points[1]))
