#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['slam_benchmark_supervisor_ros'],
 package_dir={'slam_benchmark_supervisor_ros': 'src/slam_benchmark_supervisor_ros'}
)

setup(**d)
