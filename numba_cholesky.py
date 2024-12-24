import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import scipy.linalg
import scipy.sparse as sparse
# import sklearn.gaussian_process.kernels as kernels
from numba.typed import List, Dict
from numba.core import types
from numba import njit
from numba import prange
import timeit

def logdet_chol(A):
    return 2 * np.sum(np.log(A.diagonal()))

def kl_div(A, B):
    n = A.shape[0]
    return 0.5 * (np.trace(np.linalg.solve(B, A)) - n + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])

def sparse_kl_div(A, L):
    n = A.shape[0]
    return 0.5 * (-logdet_chol(L) - np.linalg.slogdet(A)[1])

@njit(cache=True)
def create_points(n):
    points = np.zeros((n * n, 2))
    for i in range(n):
        for j in range(n):
            perturbation = np.random.uniform(-0.2, 0.2, 2)
            points[i * n + j] = np.array([i - n/2, j - n/2]) + perturbation
    return points

@njit(cache=True)
def kernel(points):
    n = points.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-np.linalg.norm(points[i] - points[j]))
    return A

def reverse_maximin(points):
    n = len(points)
    indices = np.zeros(n, dtype=int)
    lengths = np.zeros(n, dtype=float)
    dists = np.zeros(n)
    dists = np.linalg.norm(points - points[0], axis=1)
    indices[-1] = 0
    lengths[-1] = np.inf
    for i in range(n - 2, -1, -1):
        k = np.argmax(dists)
        indices[i] = k
        lengths[i] = dists[k]
        dists = np.minimum(dists, np.linalg.norm(points[k] - points, axis=1))
    return indices, lengths

def kd_sparsity_pattern(points, lengths, rho):
    tree = KDTree(points)
    near = tree.query_ball_point(points, rho * lengths)
    sparsity = List()
    for i in range(len(points)):
        l = List()
        for j in near[i]:
            if j >= i:
                l.append(j)
        sparsity.append(l)
    return sparsity

@njit(cache=True)
def col(theta):
    m = np.linalg.inv(theta)
    return m[:, 0] / np.sqrt(m[0, 0])
    

@njit(parallel=True, cache=True)
def chol(points, ptr, sparsity):
    n = len(sparsity)
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for i in prange(n):
        s = np.array(sorted(sparsity[i]))
        theta = np.zeros((len(s), 2))
        for j in range(len(s)):
            theta[j] = points[s[j]]
        c = col(kernel(theta))
        data[ptr[i] : ptr[i + 1]] = c
        indices[ptr[i] : ptr[i + 1]] = s
    return data, indices

def naive_kl_cholesky(points, rho):
    n = len(points)
    indices, lengths = reverse_maximin(points)
    ordered_points = points[indices]
    sparsity = kd_sparsity_pattern(ordered_points, lengths, rho)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = chol(ordered_points, ptr, sparsity)
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def plot(A):
    plt.matshow(np.log10(np.abs(A)), vmin=-12, vmax=1)
    plt.show()

def main():
    seed = 0
    np.random.seed(seed)
    n = 200
    points = create_points(n)
    start = timeit.default_timer()
    L = naive_kl_cholesky(points, 2.5)
    print(timeit.default_timer() - start)

if __name__ == '__main__':
    main()