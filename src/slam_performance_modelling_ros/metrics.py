#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import glob
import multiprocessing
import os
import random
import sys
import time
import traceback
from os import path
import yaml
from performance_modelling_py.environment.ground_truth_map import GroundTruthMap

from performance_modelling_py.utils import print_info, print_error
from performance_modelling_py.metrics.localization_metrics import trajectory_length_metric, absolute_localization_error_metrics, relative_localization_error_metrics_carmen_dataset, \
    estimated_pose_trajectory_length_metric, relative_localization_error_metrics_for_each_waypoint, geometric_similarity_environment_metric_for_each_waypoint
from performance_modelling_py.metrics.computation_metrics import cpu_and_memory_usage_metrics
# from performance_modelling_py.visualisation.trajectory_visualisation import save_trajectories_plot


def compute_metrics(run_output_folder, recompute_all_metrics=False):

    run_info_path = path.join(run_output_folder, "run_info.yaml")
    if not path.exists(run_info_path) or not path.isfile(run_info_path):
        print_error("run info file does not exists")
        return

    with open(run_info_path) as run_info_file:
        run_info = yaml.safe_load(run_info_file)

    environment_folder = run_info['environment_folder']
    benchmark_configuration = run_info['benchmark_configuration'] if 'benchmark_configuration' in run_info else None
    environment_type = benchmark_configuration['environment_type'] if benchmark_configuration is not None and 'environment_type' in benchmark_configuration else 'simulation'
    recorded_data_relations_path = path.join(environment_folder, "data", "recorded_data", "relations")
    ground_truth_map_info_path = path.join(environment_folder, "data", "map.yaml")
    ground_truth_map = GroundTruthMap(ground_truth_map_info_path)

    # input files
    estimated_poses_path = path.join(run_output_folder, "benchmark_data", "estimated_poses.csv")
    ground_truth_poses_path = path.join(run_output_folder, "benchmark_data", "ground_truth_poses.csv")
    scans_file_path = path.join(run_output_folder, "benchmark_data", "scans.csv")
    scans_gt_file_path = path.join(run_output_folder, "benchmark_data", "scans_gt.csv")
    run_events_file_path = path.join(run_output_folder, "benchmark_data", "run_events.csv")

    # output files
    logs_folder_path = path.join(run_output_folder, "logs")
    metrics_result_folder_path = path.join(run_output_folder, "metric_results")
    metrics_result_file_path = path.join(metrics_result_folder_path, "metrics.yaml")
    geometric_similarity_file_path = path.join(metrics_result_folder_path, "geometric_similarity.csv")
    if not path.exists(metrics_result_folder_path):
        os.makedirs(metrics_result_folder_path)

    if path.exists(metrics_result_file_path):
        with open(metrics_result_file_path) as previous_metrics_result_file:
            metrics_result_dict = yaml.safe_load(previous_metrics_result_file)
    else:
        metrics_result_dict = dict()

    # geometric_similarity
    if recompute_all_metrics or 'geometric_similarity' not in metrics_result_dict:
        if environment_type == 'simulation':
            print_info("geometric_similarity (simulation) {}".format(run_output_folder))
            metrics_result_dict['geometric_similarity'] = geometric_similarity_environment_metric_for_each_waypoint(path.join(logs_folder_path, "geometric_similarity"), geometric_similarity_file_path, scans_gt_file_path, run_events_file_path, recompute=recompute_all_metrics)

    # absolute_error_vs_scan_range(estimated_poses_path, ground_truth_poses_path, scans_file_path)

    # trajectory_length
    if recompute_all_metrics or 'trajectory_length' not in metrics_result_dict:
        if environment_type == 'simulation':
            print_info("trajectory_length (simulation) {}".format(run_output_folder))
            metrics_result_dict['trajectory_length'] = trajectory_length_metric(ground_truth_poses_path)

        if environment_type == 'dataset':
            print_info("trajectory_length (dataset) {}".format(run_output_folder))
            metrics_result_dict['trajectory_length'] = estimated_pose_trajectory_length_metric(estimated_poses_path)

    # relative_localization_error
    if recompute_all_metrics or 'relative_localization_error' not in metrics_result_dict:
        if environment_type == 'simulation':
            print_info("relative_localization_error (simulation) {}".format(run_output_folder))
            metrics_result_dict['relative_localization_error'] = relative_localization_error_metrics_for_each_waypoint(path.join(logs_folder_path, "relative_localisation_error"), estimated_poses_path, ground_truth_poses_path, run_events_file_path)

        if environment_type == 'dataset':
            print_info("relative_localization_error (dataset) {}".format(run_output_folder))
            metrics_result_dict['relative_localization_error'] = relative_localization_error_metrics_carmen_dataset(path.join(logs_folder_path, "relative_localisation_error_carmen_dataset"), estimated_poses_path, recorded_data_relations_path)

    # absolute_localization_error
    if recompute_all_metrics or 'absolute_localization_error' not in metrics_result_dict:
        if environment_type == 'simulation':
            print_info("absolute_localization_error (simulation) {}".format(run_output_folder))
            metrics_result_dict['absolute_localization_error'] = absolute_localization_error_metrics(estimated_poses_path, ground_truth_poses_path)

    # cpu_and_memory_usage
    if recompute_all_metrics or 'cpu_and_memory_usage' not in metrics_result_dict:
        print_info("cpu_and_memory_usage {}".format(run_output_folder))
        ps_snapshots_folder_path = path.join(run_output_folder, "benchmark_data", "ps_snapshots")
        metrics_result_dict['cpu_and_memory_usage'] = cpu_and_memory_usage_metrics(ps_snapshots_folder_path)

    # write metrics
    with open(metrics_result_file_path, 'w') as metrics_result_file:
        yaml.dump(metrics_result_dict, metrics_result_file, default_flow_style=False)


def parallel_compute_metrics(run_output_folder, recompute_all_metrics):
    print_info("start : compute_metrics {:3d}% {}".format(int((shared_progress.value + 1)*100/shared_num_runs.value), run_output_folder))

    # noinspection PyBroadException
    try:
        compute_metrics(run_output_folder, recompute_all_metrics=recompute_all_metrics)
    except KeyboardInterrupt:
        print_info("parallel_compute_metrics: metrics computation interrupted (run {})".format(run_output_folder))
        sys.exit()
    except:
        print_error("parallel_compute_metrics: failed metrics computation for run {}".format(run_output_folder))
        print_error(traceback.format_exc())

    shared_progress.value += 1
    print_info("finish: compute_metrics {:3d}% {}".format(int(shared_progress.value*100/shared_num_runs.value), run_output_folder))


if __name__ == '__main__':
    print_info("Python version:", sys.version_info)
    default_base_run_folder = "~/ds/performance_modelling/output/test_slam/*"
    default_num_parallel_threads = 4
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Add software version information to all run output folders.')

    parser.add_argument('--recompute', dest='recompute_all_metrics',
                        help='Whether to recompute all metrics. Defaults is false.'.format(default_base_run_folder),
                        action='store_true',
                        required=False)

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed. Defaults to {}.'.format(default_base_run_folder),
                        type=str,
                        default=default_base_run_folder,
                        required=False)

    parser.add_argument('-j', dest='num_parallel_threads',
                        help='Number of parallel threads. Defaults to {}.'.format(default_num_parallel_threads),
                        type=int,
                        default=default_num_parallel_threads,
                        required=False)

    args = parser.parse_args()

    run_folders = list(filter(path.isdir, glob.glob(path.expanduser(args.base_run_folder))))
    num_runs = len(run_folders)

    if len(run_folders) == 0:
        print_info("run folder list is empty")

    shared_progress = multiprocessing.Value('i', 0)
    shared_num_runs = multiprocessing.Value('i', len(run_folders))
    with multiprocessing.Pool(processes=args.num_parallel_threads) as pool:
        pool.starmap(parallel_compute_metrics, zip(run_folders, [args.recompute_all_metrics]*num_runs))

    # for progress, run_folder in enumerate(run_folders):
    #     print_info("main: compute_metrics {:3d}% {}".format(int((progress + 1)*100/len(run_folders)), run_folder))
    #     # noinspection PyBroadException
    #     try:
    #         compute_metrics(run_folder, recompute_all_metrics=args.recompute_all_metrics)
    #     except KeyboardInterrupt:
    #         print_info("\nmain: metrics computation interrupted")
    #         break
    #     except:
    #         print_error("failed metrics computation")
    #         print_error(traceback.format_exc())
    print_info("main: done")
