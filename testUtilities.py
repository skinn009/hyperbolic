from utilities import *


def testMinkowskiDot():
    """ Unit test for minkowskiDot"""
    print('Testing MinkowskiDot')
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([1, 1, 1, 5])
    return -14 == minkowskiDot(v1, v2)


def testGeneratePoints():
    """ Unit test for generatePoints. All generated points should lie on the H^n hyperboloid sheet:
    H^n = {x in R^{n:1} | <x, x>_{n:1} = -1, x_{n+1} > 0 }
    """
    print('Testing generatePoints')
    gen_points = generatePoints(100)
    for p in gen_points:
        # since minkowski dot should be -1, if we add 1 then we should be within 1e-5 of 0
        inner_prod = abs(minkowskiDot(p, p)+1)
        if inner_prod > 1e-5 or p[-1] <= 0:
            print('Error:', inner_prod, p)
            return False
    return True


def testMinkowskiArrayDot():
    """ Unit test for minkowskiArrayDot """
    print('Testing minkowskiArrayDot')
    vec = np.array([1, 2, 3, 4])
    arr = np.array([[1, 1, 1, 5],
                   [1, 1, 1, 6]])
    sol = np.array([[-14], [-18]])
    res = minkowskiArrayDot(arr, vec)
    return np.array_equal(res, sol)


def main():
    # test functions
    print(testMinkowskiDot())
    print(testMinkowskiArrayDot())
    print(testGeneratePoints())

    # old tests from utilities.py
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
    plot_poincare(poincare_points, save_name='plots/poincare_sample.png')

    # Three points in poincare space. B and C should be same distance away from the origin.
    # Furthermore, B and C should be further apart from each other than they are from the origin. i.e. the geodesic
    # from B->C should not be the straight path, it must arc towards the origin.
    A = np.array([0., 0.])
    B = np.array([0., 0.5])
    C = np.array([-0.5, 0.])
    print(poincareDist(A, B), poincareDist(A, C), poincareDist(B, C))


if __name__ == '__main__':
    main()