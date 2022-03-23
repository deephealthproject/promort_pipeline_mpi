# Copyright 2021-2 CRS4
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from setuptools import setup
from distutils.core import Extension
import pybind11
from glob import glob

EXTRA_COMPILE_ARGS =['-fvisibility=hidden', '-g0', '-Wall', '-Wextra', '-pedantic', '-std=c++17', '-O2']

cpp_handler = Extension("OPT_MPI",
        sorted(glob("*/*.cpp")),
        include_dirs=[
            '/usr/local/include/eigen3/',
            pybind11.get_include(user=True),
        ],
        language='c++',
        libraries=['eddl', 'cudart', 'mpi', 'ecvl_eddl'],
        library_dirs=[
            '/usr/local/cuda/lib64/',
        ],
        extra_compile_args=EXTRA_COMPILE_ARGS,  
    )

ext_mods = [ cpp_handler ]

setup(
    name="opt_mpi",
    version="0.1",
    author="Giovanni Busonera, Francesco Versaci",
    author_email="giovanni.busonera@crs4.it, francesco.versaci@gmail.com",
    description="MPI extensions for DeepHealth SGD optimizer",
    ext_modules=ext_mods,
    python_requires='>=3.6',
)
