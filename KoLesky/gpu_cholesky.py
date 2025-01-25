import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import tree_util
import scipy.sparse as sparse
import jax
import timeit

def create_points(n):
    seed = 0
    np.random.seed(seed)
    points = np.zeros((n * n, 2))
    for i in range(n):
        for j in range(n):
            perturbation = np.random.uniform(-0.2, 0.2, 2)
            points[i * n + j] = np.array([i - n/2, j - n/2]) + perturbation
    return points

def logdet_chol(A):
    return 2 * np.sum(np.log(A.diagonal()))

def kl_div(A, B):
    n = A.shape[0]
    return 0.5 * (np.trace(np.linalg.solve(B, A)) - n + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])

def sparse_kl_div(A, L):
    n = A.shape[0]
    return 0.5 * (-logdet_chol(L) - np.linalg.slogdet(A)[1])

@jit
def kernel(points):
    n = points.shape[0]
    A = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A = A.at[i, j].set(jnp.exp(-jnp.linalg.norm(points[i] - points[j])))
    return A

def naive_sparsity_pattern(points, lengths, rho):
    n = len(points)
    sparsity = {i : [] for i in range(n)}
    for i in range(n):
        for j in range(i, n):
            if np.linalg.norm(points[i] - points[j]) <= max(lengths[i], lengths[j]) * rho:
                sparsity[i].append(j)
        sparsity[i] = jnp.array(sparsity[i])
    return sparsity

def reverse_maximin(points):
    n = np.shape(points)[0]
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

@jit
def col(theta):
    m = jnp.linalg.inv(theta)
    return m[:, 0] / jnp.sqrt(m[0, 0])

# def chol(points, sparsity):
#     n = len(points)
#     ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
#     data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
#     for i in range(n):
#         s = sorted(sparsity[i])
#         c = col(kernel(points[s]))
#         data[ptr[i] : ptr[i + 1]] = c
#         indices[ptr[i] : ptr[i + 1]] = s
#     return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

# def naive_kl_cholesky(points, rho):
#     n = len(points)
#     indices, lengths = reverse_maximin(points)
#     ordered_points = points[indices]
#     sparsity = naive_sparsity_pattern(ordered_points, lengths, rho)
#     return chol(ordered_points, sparsity)

@jit
def chol(points, sparsity, data, indices, ptr):
    n = len(points)
    flattened, tree_structure = tree_util.tree_flatten(sparsity)
    idx = 0
    for i in range(n):
        s = flattened[i]
        c = col(kernel(points[s]))
        for j in range(len(c)):
            data = data.at[idx].set(c[j])
            indices = indices.at[idx].set(s[j])
            idx += 1
    return data, indices

def naive_kl_cholesky(points, rho):
    n = len(points)
    indices, lengths = reverse_maximin(points)
    ordered_points = points[indices]
    sparsity = naive_sparsity_pattern(ordered_points, lengths, rho)
    ptr = jnp.cumsum(jnp.array([0] + [len(sparsity[i]) for i in range(n)]))
    data, indices = jnp.zeros(ptr[-1], dtype=jnp.float32), jnp.zeros(ptr[-1], dtype=jnp.int32)
    data, indices = chol(ordered_points, sparsity, data, indices, ptr)
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def main():
    n = 4
    points = create_points(n)
    order, lengths = reverse_maximin(points)
    ordered_points = points[order]
    print(jax.devices())
    start = timeit.default_timer()
    L = naive_kl_cholesky(points, 1.8).toarray()
    print(timeit.default_timer() - start)

if __name__ == "__main__":
    main()