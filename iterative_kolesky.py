import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import scipy.linalg
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels
import timeit
from KoLesky.maxheap import Heap
from KoLesky.ordering import reverse_maximin
from KoLesky.ordering import sparsity_pattern
from KoLesky.ordering import update_dists
def logdet_chol(A):
    return 2 * np.sum(np.log(A.diagonal()))

def kl_div(A, B):
    n = A.shape[0]
    return 0.5 * (np.trace(np.linalg.solve(B, A)) - n + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])

def sparse_kl_div(A, L):
    n = A.shape[0]
    return 0.5 * (-logdet_chol(L) - np.linalg.slogdet(A)[1])

# def kernel(points):
#     n = points.shape[0]
#     A = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             A[i, j] = np.exp(-np.linalg.norm(points[i] - points[j]))
#     return A

def new_kernel_idx(points, kernel, L, r, c):
    # calculate L.T @ theta @ L index at r and c
    left_row = L.T[r]
    right_col = L[:, c]
    left_indices = left_row.indices
    right_indices = right_col.indices
    left_points = points[left_indices]
    right_points = points[right_indices]
    kernel_matrix = kernel(left_points, right_points)
    result = np.dot(left_row.data, np.dot(kernel_matrix, right_col.data))
    return result
    

def naive_reverse_maximin(points):
    n = np.shape(points)[0]
    indices = np.zeros(n, dtype=int)
    lengths = np.zeros(n, dtype=float)
    dists = np.linalg.norm(points - points[0], axis=1)
    indices[-1] = 0
    lengths[0] = np.inf
    for i in range(n - 2, -1, -1):
        k = np.argmax(dists)
        indices[i] = k
        lengths[i] = dists[k]
        dists = np.minimum(dists, np.linalg.norm(points[k] - points, axis=1))
    return indices, lengths

def py_reverse_maximin(points):
    n = len(points)
    indices = np.zeros(n, dtype=np.int64)
    lengths = np.zeros(n)
    tree = KDTree(points)
    dists = np.linalg.norm(points - points[0], axis=1)
    indices = np.zeros(n, dtype=np.int64)
    lengths = np.zeros(n)
    heap = Heap(dists, np.arange(n))
    for i in range(n - 1, -1, -1):
        lk, k = heap.pop()
        indices[i] = k
        lengths[i] = lk
        js = tree.query_ball_point(points[k], lk)
        dists = np.linalg.norm(points[js] - points[k], axis=1)
        for index, j in enumerate(js):
            heap.decrease_key(j, dists[index])
    return indices, lengths

def py_p_reverse_maximin(points, p = 1):
    inf = 1e6
    n = len(points)
    indices = np.zeros(n, dtype = np.int64)
    lengths = np.zeros(n)
    dists = np.array([[-i + inf] * p for i in range(n)])
    tree = KDTree(points)
    heap = Heap(np.max(dists, axis=1), np.arrange(n))
    for i in range(n - 1, -1, -1):
        lk, k = heap.pop()
        indices[i] = k
        lengths[i] = lk if lk < inf - n else np.inf
        js = tree.query_ball_point(points[k], lk)
        dists_k = np.linalg.norm(points[js] - points[k], axis=1)
        update_dists(heap, dists, dists_k, np.array(js, dtype=np.int64))
    return indices, lengths
        
def naive_sparsity_pattern(points, lengths, rho):
    n = len(points)
    sparsity = {i : [] for i in range(n)}
    for i in range(n):
        for j in range(i, n):
            if np.linalg.norm(points[i] - points[j]) <= max(lengths[i], lengths[j]) * rho:
                sparsity[i].append(j)
    return sparsity

def kd_sparsity_pattern(points, lengths, rho):
    tree = KDTree(points)
    near = tree.query_ball_point(points, rho * lengths)
    return {i: [j for j in near[i] if j >= i] for i in range(len(points))}

def py_sparsity_pattern(points, lengths, rho):
    tree, offset, length_scale = KDTree(points), 0, lengths[0]
    sparsity = {}
    for i in range(len(points)):
        if lengths[i] > 2 * length_scale:
            tree, offset, length_scale = KDTree(points[i:]), i, lengths[i]
        sparsity[i] = [
            offset + j
            for j in tree.query_ball_point(points[i], rho * lengths[i])
            if offset + j >= i
        ]
    return sparsity

def supernodes(sparsity, lengths, lamb):
    groups = []
    candidates = set(range(len(lengths)))
    agg_sparsity = {}
    i = 0
    while len(candidates) > 0:
        while i not in candidates:
            i += 1
        group = sorted(j for j in sparsity[i] if lengths[j] <= lamb * lengths[i] and j in candidates)
        groups.append(group)
        candidates -= set(group)
        s = sorted({k for j in group for k in sparsity[j]})
        agg_sparsity[group[0]] = s
        positions = {k: j for j, k in enumerate(s)}
        for j in group[1:]:
            agg_sparsity[j] = np.empty(len(s) - positions[j], dtype=int)
    return groups, agg_sparsity

def col(theta):
    m = np.linalg.inv(theta)
    return m[:, 0] / np.sqrt(m[0, 0])

def cols(theta):
    return np.flip(np.linalg.cholesky(np.flip(theta))).T

def chol(points, kernel, sparsity):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for i in range(n):
        s = sorted(sparsity[i])
        c = col(kernel(points[s]))
        data[ptr[i] : ptr[i + 1]] = c
        indices[ptr[i] : ptr[i + 1]] = s
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def aggregate_chol(points, kernel, sparsity, groups):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for group in groups:
        s = sorted(sparsity[group[0]])
        positions = {i: k for k, i in enumerate(s)}
        L_group = cols(kernel(points[s]))
        for i in group:
            k = positions[i]
            e_k = np.zeros(len(s))
            e_k[k] = 1
            col = scipy.linalg.solve_triangular(L_group, e_k, lower=True, check_finite=False)
            data[ptr[i] : ptr[i + 1]] = col[k:]
            indices[ptr[i] : ptr[i + 1]] = s[k:]
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def iter_col(points, kernel, s, L, calculated_entries):
    m = np.zeros((len(s), len(s)))
    for i in range(len(s)):
        for j in range(len(s)):
            if (i, j) in calculated_entries: 
                m[i, j] = calculated_entries[(i, j)]
            elif (j, i) in calculated_entries: 
                m[i, j] = calculated_entries[(j, i)]
            else: 
                m[i, j] = new_kernel_idx(points, kernel, L, s[i], s[j])
                calculated_entries[(i, j)] = m[i, j]
    return col(m)

def iter_chol(points, kernel, sparsity, L):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for i in range(n):
        s = sorted(sparsity[i])
        entries = {}
        c = iter_col(points, kernel, s, L, entries)
        data[ptr[i] : ptr[i + 1]] = c
        indices[ptr[i] : ptr[i + 1]] = s
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def iter_cols(points, kernel, sparsity, L):
    m = np.zeros((len(sparsity), len(sparsity)))
    for i in range(len(sparsity)):
        for j in range(len(sparsity)):
            m[i, j] = new_kernel_idx(points, kernel, L, i, j)
    return cols(m)

def aggregate_iter_chol(points, kernel, sparsity, groups, L):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for group in groups:
        s = sorted(sparsity[group[0]])
        positions = {i: k for k, i in enumerate(s)}
        L_group = iter_cols(points, kernel, s, L)
        for i in group:
            k = positions[i]
            e_k = np.zeros(len(s))
            e_k[k] = 1
            col = scipy.linalg.solve_triangular(L_group, e_k, lower=True, check_finite=False)
            data[ptr[i] : ptr[i + 1]] = col[k:]
            indices[ptr[i] : ptr[i + 1]] = s[k:]
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def naive_kl_cholesky(points, rho):
    indices, lengths = reverse_maximin(points)
    ordered_points = points[indices]
    sparsity = sparsity_pattern(ordered_points, lengths, rho)
    kernel = kernels.Matern(length_scale=1.0, nu=0.5)
    return chol(ordered_points, kernel, sparsity)

def aggregated_kl_cholesky(points, rho, lamb):
    indices, lengths = reverse_maximin(points)
    ordered_points = points[indices]
    sparsity = sparsity_pattern(ordered_points, lengths, rho)
    groups, agg_sparsity = supernodes(sparsity, lengths, lamb)
    kernel = kernels.Matern(length_scale=1.0, nu=0.5)
    return aggregate_chol(ordered_points, kernel, agg_sparsity, groups)

def naive_iterative_kl_cholesky(points, rho):
    indices, lengths = reverse_maximin(points)
    ordered_points = points[indices]
    sparsity = naive_sparsity_pattern(ordered_points, lengths, rho)
    kernel = kernels.Matern(length_scale=1.0, nu=0.5)
    L = chol(ordered_points, kernel, sparsity)
    iter_L = iter_chol(ordered_points, kernel, sparsity, L)
    return L, iter_L

def iterative_aggregated_kl_cholesky(points, rho, lamb):
    indices, lengths = reverse_maximin(points)
    ordered_points = points[indices]
    sparsity = naive_sparsity_pattern(ordered_points, lengths, rho)
    groups, agg_sparsity = supernodes(sparsity, lengths, lamb)
    kernel = kernels.Matern(length_scale=1.0, nu=0.5)
    theta = kernel(ordered_points)
    L = aggregate_chol(theta, agg_sparsity, groups)
    new_kernel = L.T @ theta @ L
    return L @ aggregate_chol(new_kernel, agg_sparsity, groups)

def plot(A):
    plt.matshow(np.log10(np.abs(A)), vmin=-12, vmax=1)
    plt.show()

def kl_plot(points):
    order, lengths = reverse_maximin(points)
    ordered_points = points[order]
    kernel = kernels.Matern(length_scale=1, nu=0.5)
    A = kernel(ordered_points)
    delta = 0.1
    for i in range(1, 20):
        rho = 1.8 + i * delta
        L = naive_iterative_kl_cholesky(points, rho)
        kl = sparse_kl_div(A, L)
        print(kl)
        plt.scatter(rho, np.log(kl), color='blue')
        L = naive_kl_cholesky(points, rho)
        kl = sparse_kl_div(A, L)
        plt.scatter(rho, np.log(kl), color='red')
        plt.xlabel("rho")
        plt.ylabel("log(KL(Theta, inv(LL^T)))")
        plt.legend(["iterative", "pevious"])
    plt.show()

def plot_column_scatterplot(A, col):
    for i in range(A.shape[1]):
        if A[i, col] != 0:
            plt.scatter(i, A[i, col], color='black')
    plt.show()

def plot_3d(points, theta, col):
    ax = plt.figure().add_subplot(projection='3d')
    mag = np.log10(np.abs(theta[:, col]))
    ax.scatter(points[:, 0], points[:, 1], mag)
    plt.show()

def main():
    seed = 0
    np.random.seed(seed)
    n = 1000
    points = np.zeros((n * n, 2))
    for i in range(n):
        for j in range(n):
            perturbation = np.random.uniform(-0.2, 0.2, 2)
            points[i * n + j] = np.array([i - n/2, j - n/2]) + perturbation
    order, lengths = reverse_maximin(points)
    ordered_points = points[order]



if __name__ == '__main__':
    main()