# cython: profile=False

import numpy as np
cimport numpy as np
from scipy.spatial import KDTree
from .maxheap cimport Heap
from . cimport mkl


np.import_array()

cdef double[:] _distance_vector(double [:, ::1] points, double [::1] point):
    cdef:
        int n, i, j
        double dist, d
        double *start
        double *p
        double[:] res

    n = points.shape[1]
    start = &points[0, 0]
    p = &point[0]
    cdef double[n] test
    res = test
    for i in range(points.shape[0]):
        dist = 0
        for j in range(n):
            d = (start + i*n)[j] - p[j]
            dist += d*d
        res[i] = dist
    mkl.vdSqrt(points.shape[0], &res, &res)
    return res

cpdef reverse_maximin(double[:, ::1] points):
    cdef:
        int n, i, k, index, j
        double lk
        int[:] indices, js
        double[:] lengths, dists
        Heap heap
    n = points.shape[0]
    indices = int[n]
    lengths = double[n]
    tree = KDTree(points)
    dists = double
    _distance_vector(points, points[0])
    heap = Heap(dists, np.arange(n))
    for i in range(n - 1, -1, -1):
        lk, k = heap.pop()
        indices[i] = k
        lengths[i] = lk
        js = tree.query_ball_point(points[k], lk)
        _distance_vector(points[js], points[k])
        # dists = np.linalg.norm(points[js] - points[k], axis=1)
        for index in range(len(js)):
            j = js[range]
            heap.decrease_key(j, dists[index])
    return indices, lengths