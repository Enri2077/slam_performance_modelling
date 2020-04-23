#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import rospkg
import argparse
from os import path

from slam_benchmark_run import BenchmarkRun
from performance_modelling_py.benchmark_execution.grid_benchmarking import execute_grid_benchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the benchmark')

    parser.add_argument('-e', dest='environment_dataset_folder',
                        help='Dataset folder containg the stage environment.world file (recursively).',
                        type=str,
                        default="~/ds/performance_modelling_few_datasets/",
                        required=False)

    parser.add_argument('-c', dest='grid_benchmark_configuration',
                        help='Yaml file with the configuration of the benchmark.',
                        type=str,
                        default="slam_grid_benchmark_1.yaml",
                        required=False)

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed.',
                        type=str,
                        default="~/ds/performance_modelling_output/test/",
                        required=False)

    parser.add_argument('-n', '--num-runs', dest='num_runs',
                        help='Number of runs to be executed for each combination of configurations.',
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument('-g', '--headless', dest='headless',
                        help='When set the components are run with no GUI.',
                        action='store_true',
                        required=False)

    parser.add_argument('-s', '--show-ros-info', dest='show_ros_info',
                        help='When set the component nodes are launched with output="screen".',
                        action='store_true',
                        required=False)

    args = parser.parse_args()
    base_run_folder = path.expanduser(args.base_run_folder)
    environment_dataset_folder = path.expanduser(args.environment_dataset_folder)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('slam_performance_modelling')
    grid_benchmark_configuration = path.join(package_path, "config", "benchmark_configurations", args.grid_benchmark_configuration)
    components_configurations_folder = path.join(package_path, "config", "component_configurations")

    environment_folders = sorted(map(path.dirname, set(glob.glob(path.join(path.abspath(path.expanduser(environment_dataset_folder)), "**/*.world"))).union(set(glob.glob(path.join(path.abspath(path.expanduser(environment_dataset_folder)), "*.world"))))))

    execute_grid_benchmark(benchmark_run_object=BenchmarkRun,
                           grid_benchmark_configuration=grid_benchmark_configuration,
                           components_configurations_folder=components_configurations_folder,
                           environment_folders=environment_folders,
                           base_run_folder=base_run_folder,
                           num_runs=args.num_runs,
                           headless=args.headless,
                           show_ros_info=args.show_ros_info)
