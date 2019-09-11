import numpy as np
import math
from scipy.stats import multivariate_normal


def AsymMOG2d(x):
    A = np.array([
        [0.5, 0.],
        [0., 0.5]
    ])
    B = np.array([
        [1., 0.],
        [0., 1.]
    ])
    C = np.array([
        [2.0, 0.],
        [0., 2.]
    ])

    mu1 = np.array([4*math.sqrt(3) - 1, 1])
    v1Av1 = (x - mu1) @ A @ (x - mu1)
    mu2 = np.array([-4*math.sqrt(3) - 1, 1])
    v2Bv2 = (x - mu2) @ B @ (x - mu2)
    mu3 = np.array([2, -10])
    v3Cv3 = (x - mu3) @ C @ (x - mu3)

    detA = np.linalg.det(np.linalg.inv(A))
    detB = np.linalg.det(np.linalg.inv(B))
    detC = np.linalg.det(np.linalg.inv(C))
    gaussian1 = np.exp(-v1Av1/2) / np.sqrt((2*math.pi)**2 * detA)
    gaussian2 = np.exp(-v2Bv2/2) / np.sqrt((2*math.pi)**2 * detB)
    gaussian3 = np.exp(-v3Cv3/2) / np.sqrt((2*math.pi)**2 * detC)
    MOG = (gaussian1 + gaussian2 + gaussian3) / 3

    return MOG


def AsymMOG2d_cdf(x):
    A = multivariate_normal.cdf(
        x, mean=np.array([4*math.sqrt(3)-1, 1]), cov=0.5)
    B = multivariate_normal.cdf(
        x, mean=np.array([-4*math.sqrt(3)-1, 1]), cov=1)
    C = multivariate_normal.cdf(x, mean=np.array([2, -10]), cov=2)
    return (A+B+C)/3
