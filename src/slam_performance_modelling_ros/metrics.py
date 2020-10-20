#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
from os import path
import yaml
from performance_modelling_py.environment.ground_truth_map import GroundTruthMap

from performance_modelling_py.utils import print_info, print_error
from performance_modelling_py.metrics.localization_metrics import trajectory_length_metric, absolute_localization_error_metrics, relative_localization_error_metrics, relative_localization_error_metrics_carmen_dataset, \
    estimated_pose_trajectory_length_metric
from performance_modelling_py.metrics.computation_metrics import cpu_and_memory_usage_metrics
# from performance_modelling_py.visualisation.trajectory_visualisation import save_trajectories_plot


def compute_metrics(run_output_folder):
    metrics_result_dict = dict()

    run_info_path = path.join(run_output_folder, "run_info.yaml")
    if not path.exists(run_info_path) or not path.isfile(run_info_path):
        print_error("run info file does not exists")

    with open(run_info_path) as run_info_file:
        run_info = yaml.safe_load(run_info_file)

    environment_folder = run_info['environment_folder']
    ground_truth_map_info_path = path.join(environment_folder, "data", "map.yaml")
    ground_truth_map = GroundTruthMap(ground_truth_map_info_path)
    recorded_data_relations_path = path.join(environment_folder, "data", "recorded_data", "relations")

    # localization metrics
    estimated_correction_poses_path = path.join(run_output_folder, "benchmark_data", "estimated_correction_poses.csv")
    estimated_poses_path = path.join(run_output_folder, "benchmark_data", "estimated_poses.csv")
    ground_truth_poses_path = path.join(run_output_folder, "benchmark_data", "ground_truth_poses.csv")
    scans_file_path = path.join(run_output_folder, "benchmark_data", "scans.csv")

    logs_folder_path = path.join(run_output_folder, "logs")

    print_info("trajectory_length")
    if path.exists(ground_truth_poses_path):
        metrics_result_dict['trajectory_length'] = trajectory_length_metric(ground_truth_poses_path)
    else:
        metrics_result_dict['trajectory_length'] = estimated_pose_trajectory_length_metric(estimated_poses_path)

    # if path.exists(ground_truth_poses_path):
    #     print_info("relative_localization_correction_error")
    #     metrics_result_dict['relative_localization_correction_error'] = relative_localization_error_metrics(path.join(logs_folder_path, "relative_localisation_correction_error"), estimated_correction_poses_path, ground_truth_poses_path)

    if path.exists(ground_truth_poses_path):
        print_info("relative_localization_error from ground truth")
        metrics_result_dict['relative_localization_error'] = relative_localization_error_metrics(path.join(logs_folder_path, "relative_localisation_error"), estimated_poses_path, ground_truth_poses_path)

    if path.exists(recorded_data_relations_path):
        print_info("relative_localization_error from carmen relations")
        metrics_result_dict['relative_localization_error'] = relative_localization_error_metrics_carmen_dataset(path.join(logs_folder_path, "relative_localisation_error_carmen_dataset"), estimated_poses_path, recorded_data_relations_path)
        print("\n\n\n", metrics_result_dict['relative_localization_error'], "\n\n\n")

    # if path.exists(ground_truth_poses_path):
    #     print_info("absolute_localization_correction_error")
    #     metrics_result_dict['absolute_localization_correction_error'] = absolute_localization_error_metrics(estimated_correction_poses_path, ground_truth_poses_path)

    if path.exists(ground_truth_poses_path):
        print_info("absolute_localization_error")
        metrics_result_dict['absolute_localization_error'] = absolute_localization_error_metrics(estimated_poses_path, ground_truth_poses_path)

    # computation metrics
    print_info("cpu_and_memory_usage")
    ps_snapshots_folder_path = path.join(run_output_folder, "benchmark_data", "ps_snapshots")
    metrics_result_dict['cpu_and_memory_usage'] = cpu_and_memory_usage_metrics(ps_snapshots_folder_path)

    # write metrics
    metrics_result_folder_path = path.join(run_output_folder, "metric_results")
    metrics_result_file_path = path.join(metrics_result_folder_path, "metrics.yaml")
    if not path.exists(metrics_result_folder_path):
        os.makedirs(metrics_result_folder_path)
    with open(metrics_result_file_path, 'w') as metrics_result_file:
        yaml.dump(metrics_result_dict, metrics_result_file, default_flow_style=False)

    # # absolute_error_vs_voronoi_radius(estimated_poses_path, ground_truth_poses_path, ground_truth_map).to_csv(path.join(metrics_result_folder_path, "abs_err_vs_voronoi_radius.csv"))
    # absolute_error_vs_scan_range(estimated_poses_path, ground_truth_poses_path, scans_file_path)

    # # visualisation
    # print_info("visualisation")
    # visualisation_output_folder = path.join(run_output_folder, "visualisation")
    # save_trajectories_plot(visualisation_output_folder, estimated_poses_path, estimated_correction_poses_path, ground_truth_poses_path)


if __name__ == '__main__':
    run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_slam_dataset/*"))))
    for progress, run_folder in enumerate(run_folders):
        print_info("main: compute_metrics {}% {}".format((progress + 1)*100/len(run_folders), run_folder))
        compute_metrics(path.expanduser(run_folder))

    # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/elysium/performance_modelling/output/localization/*"))))
    # # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_localization/*"))))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    # compute_metrics(last_run_folder)
