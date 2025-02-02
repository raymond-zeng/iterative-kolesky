import numpy as np
cimport numpy as np
from scipy.spatial import KDTree
from .maxheap cimport Heap
from . cimport mkl

np.import_array()
cdef double[::1] _distance_vector(double[:, ::1] points, double[::1] point):
   cdef:
       int n, i, j
       double dist, d
       double *start
       double *p
       double[::1] dists
   n = points.shape[1]
   start = &points[0, 0]
   p = &point[0]
   dists = np.empty(points.shape[0], np.float64)
   for i in range(points.shape[0]):
       dist = 0
       for j in range(n):
           d = (start + i*n)[j] - p[j]
           dist += d*d
       dists[i] = dist
   mkl.vdSqrt(points.shape[0], &dists[0], &dists[0])
   return dists

cpdef tuple reverse_maximin(np.ndarray[np.float64_t, ndim=2] points):
    cdef:
        int n, i, k, index, j
        double lk
        long[::1] indices
        double[::1] lengths
        Heap heap
        double[:, ::1] points_js
        double[::1] dists
        list js
    n = points.shape[0]
    indices = np.empty(n, dtype = np.long)
    lengths = np.empty(n, dtype = np.float64)
    tree = KDTree(points)
    dists = _distance_vector(points, points[0])
    heap = Heap(dists, np.arange(n))
    for i in range(n - 1, -1, -1):
        lk, k = heap.pop()
        indices[i] = k
        lengths[i] = lk
        js = tree.query_ball_point(points[k], lk)
        dists = _distance_vector(points[js], points[k])
        for index in range(len(js)):
            j = js[index]
            heap.decrease_key(j, dists[index])
    return indices, lengths

cpdef object[::1] sparsity_pattern(double[:, ::1] points, double[::1] lengths, double rho):
    cdef:
        int n, i, offset
        long j
        double length_scale
        object[::1] sparsity
    n = points.shape[0]
    tree = KDTree(points)
    offset = 0
    length_scale = lengths[0]
    sparsity = np.empty(n, dtype = object)
    for i in range(n):
        if lengths[i] > 2 * length_scale:
            tree = KDTree(points[i:])
            offset = i
            length_scale = lengths[i]
        sparsity[i] = [
            offset + j
            for j in tree.query_ball_point(points[i], rho * lengths[i])
            if offset + j >= i
        ]
    return sparsity