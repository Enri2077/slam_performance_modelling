#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import argparse
from os import path

from slam_performance_modelling_ros.slam_benchmark_run import BenchmarkRun
from performance_modelling_py.benchmark_execution.grid_benchmarking import execute_grid_benchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the benchmark')

    parser.add_argument('-e', dest='environment_dataset_folders',
                        help='Dataset folders containg the environment data. Use wildcards to select multiple folders. Only folders are selected, files are ignored.',
                        type=str,
                        default="~/ds/performance_modelling/test_datasets/dataset/*",
                        required=False)

    parser.add_argument('-c', dest='grid_benchmark_configuration',
                        help='Yaml file with the configuration of the benchmark.',
                        type=str,
                        default="~/w/catkin_ws/src/slam_performance_modelling/config/benchmark_configurations/slam_grid_benchmark_slam_toolbox_experimental_nav_config.yaml",
                        required=False)

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed.',
                        type=str,
                        default="~/ds/performance_modelling/output/test_slam/",
                        required=False)

    parser.add_argument('-n', '--num-runs', dest='num_runs',
                        help='Number of runs to be executed for each combination of configurations.',
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument('--gui', dest='gui',
                        help='When set the components are run with no GUI.',
                        action='store_true',
                        required=False)

    parser.add_argument('-s', '--show-ros-info', dest='show_ros_info',
                        help='When set the component nodes are launched with output="screen".',
                        action='store_true',
                        required=False)

    args = parser.parse_args()
    base_run_folder = path.expanduser(args.base_run_folder)
    environment_folders = sorted(filter(path.isdir, glob.glob(path.expanduser(args.environment_dataset_folders))))
    grid_benchmark_configuration = path.expanduser(args.grid_benchmark_configuration)

    execute_grid_benchmark(benchmark_run_object=BenchmarkRun,
                           grid_benchmark_configuration=grid_benchmark_configuration,
                           environment_folders=environment_folders,
                           base_run_folder=base_run_folder,
                           num_runs=args.num_runs,
                           headless=not args.gui,
                           show_ros_info=args.show_ros_info)
