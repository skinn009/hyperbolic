
import numpy as np
import math

def generate_points (N):
    """
    We generate N points on the hyperboloid satisfying the equation . We get random numbers for x1, x2, and compute
    for x3 where x3 = sqrt(1 = x1^2 + x2^2).
    :param N:
    :return:
    """
    rands = [(np.random.rand(), np.random.rand()) for i in range(N)]
    point_list = [(x1, x2, math.sqrt(1 + x1**2 + x2**2)) for (x1,x2) in rands]
    return point_list





if __name__ == "__main__":
    print(generate_points(10))

