from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

path = Path(__file__).parent

include_dirs = [
    str(path / "include"),
    str(path / ".venv/include"),
    np.get_include(),
]
# library_dirs = [
#     str(path / "lib"),
#     str(path / ".venv/Lib"),
# ]
# https://www.hlibpro.com/doc/3.1/install.html
# https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html
libraries = [
    "hpro",
    "lapack",
    "blas",
    "boost_filesystem",
    "boost_system",
    "boost_program_options",
    "boost_iostreams",
    "tbb",
    "z",
    "metis",
    "fftw3",
    "gsl",
    "gslcblas",
    "m",
    "hdf5",
    "hdf5_cpp",
    "m",
    "stdc++",
] + [
    "mkl_intel_lp64",
    "mkl_intel_thread",
    "mkl_core",
    "iomp5",
    "pthread",
    "m",
    "dl",
]
extra_compile_args = ["-Wall", "-O3", "-fopenmp", "-fPIC"]

rpath = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.42.34433/lib/x64'

extensions = [
    Extension(
        "*",
        ["KoLesky/*.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=include_dirs,
        # library_dirs=library_dirs,
        libraries=libraries,
        # runtime_library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=[f'/LIBPATH:{rpath}']
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)