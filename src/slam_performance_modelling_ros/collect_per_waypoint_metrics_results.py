#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
print("Python version:", sys.version_info)
if sys.version_info.major < 3:
    print("Python version less than 3")
    sys.exit()

import glob
import argparse
import pickle
import yaml
import pandas as pd
from os import path

from performance_modelling_py.utils import print_info, print_error


def get_simple_value(result_path):
    with open(result_path) as result_file:
        return result_file.read()


def get_csv(result_path):
    df_csv = pd.read_csv(result_path, sep=', ', engine='python')
    return df_csv


def get_yaml(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        return yaml.load(yaml_file)


def get_yaml_by_path(yaml_dict, keys):
    assert(isinstance(keys, list))
    try:
        if len(keys) > 1:
            return get_yaml_by_path(yaml_dict[keys[0]], keys[1:])
        elif len(keys) == 1:
            return yaml_dict[keys[0]]
        else:
            return None
    except (KeyError, TypeError):
        return None


def collect_data(base_run_folder_path, invalidate_cache=False):

    base_run_folder = path.expanduser(base_run_folder_path)
    cache_file_path = path.join(base_run_folder, "run_data_per_waypoint_cache.pkl")

    if not path.isdir(base_run_folder):
        print_error("collect_data: base_run_folder does not exists or is not a directory".format(base_run_folder))
        return None, None

    def is_completed_run_folder(p):
        return path.isdir(p) and path.exists(path.join(p, "benchmark_data.bag"))

    run_folders = sorted(list(filter(is_completed_run_folder, glob.glob(path.abspath(base_run_folder) + '/*'))))

    record_list = list()
    if invalidate_cache or not path.exists(cache_file_path):
        df = pd.DataFrame()
        parameter_names = set()
        cached_run_folders = set()
    else:
        print_info("collect_data: updating cache")
        with open(cache_file_path, 'rb') as f:
            cache = pickle.load(f)
        df = cache['df']
        parameter_names = cache['parameter_names']
        cached_run_folders = set(df['run_folder'])

    # collect results from runs not already cached
    print_info("collect_data: reading run data")
    no_output = True
    for i, run_folder in enumerate(run_folders):
        metric_results_folder = path.join(run_folder, "metric_results")
        run_info_file_path = path.join(run_folder, "run_info.yaml")
        metrics_file_path = path.join(metric_results_folder, "metrics.yaml")

        if not path.exists(metric_results_folder):
            continue
        if not path.exists(run_info_file_path):
            print_error("collect_data: run_info file does not exists [{}]".format(run_info_file_path))
            no_output = False
            continue
        if run_folder in cached_run_folders:
            continue

        run_info = get_yaml(run_info_file_path)

        try:
            metrics_dict = get_yaml(metrics_file_path)
        except IOError:
            print_error("metric_results could not be read: {}".format(metrics_file_path))
            no_output = False
            continue

        run_record = dict()

        for parameter_name, parameter_value in run_info['run_parameters'].items():
            parameter_names.add(parameter_name)
            if type(parameter_value) == list:
                parameter_value = tuple(parameter_value)
            run_record[parameter_name] = parameter_value

        parameter_names.add('environment_name')
        run_record['environment_name'] = path.basename(path.abspath(run_info['environment_folder']))
        run_record['run_folder'] = run_folder

        # collect per waypoint metric results
        waypoint_start_times = set()

        geometric_similarity_per_waypoint_dict = dict()
        geometric_similarity_per_waypoint_list = get_yaml_by_path(metrics_dict, ['geometric_similarity', 'geometric_similarity_per_waypoint_list'])
        if geometric_similarity_per_waypoint_list is not None:
            for geometric_similarity_per_waypoint in geometric_similarity_per_waypoint_list:
                if geometric_similarity_per_waypoint is not None and 'start_time' in geometric_similarity_per_waypoint:
                    geometric_similarity_per_waypoint_dict[geometric_similarity_per_waypoint['start_time']] = geometric_similarity_per_waypoint
                    waypoint_start_times.add(geometric_similarity_per_waypoint['start_time'])

        geometric_similarity_range_limit_per_waypoint_dict = dict()
        geometric_similarity_range_limit_per_waypoint_list = get_yaml_by_path(metrics_dict, ['geometric_similarity_range_limit', 'geometric_similarity_per_waypoint_list'])
        if geometric_similarity_range_limit_per_waypoint_list is not None:
            for geometric_similarity_range_limit_per_waypoint in geometric_similarity_range_limit_per_waypoint_list:
                if geometric_similarity_range_limit_per_waypoint is not None and 'start_time' in geometric_similarity_range_limit_per_waypoint:
                    geometric_similarity_range_limit_per_waypoint_dict[geometric_similarity_range_limit_per_waypoint['start_time']] = geometric_similarity_range_limit_per_waypoint
                    waypoint_start_times.add(geometric_similarity_range_limit_per_waypoint['start_time'])

        geometric_similarity_sensor_per_waypoint_dict = dict()
        geometric_similarity_sensor_per_waypoint_list = get_yaml_by_path(metrics_dict, ['geometric_similarity_sensor', 'geometric_similarity_per_waypoint_list'])
        if geometric_similarity_sensor_per_waypoint_list is not None:
            for geometric_similarity_sensor_per_waypoint in geometric_similarity_sensor_per_waypoint_list:
                if geometric_similarity_sensor_per_waypoint is not None and 'start_time' in geometric_similarity_sensor_per_waypoint:
                    geometric_similarity_sensor_per_waypoint_dict[geometric_similarity_sensor_per_waypoint['start_time']] = geometric_similarity_sensor_per_waypoint
                    waypoint_start_times.add(geometric_similarity_sensor_per_waypoint['start_time'])

        lidar_visibility_per_waypoint_dict = dict()
        lidar_visibility_per_waypoint_list = get_yaml_by_path(metrics_dict, ['lidar_visibility', 'lidar_visibility_per_waypoint_list'])
        if lidar_visibility_per_waypoint_list is not None:
            for lidar_visibility_per_waypoint in lidar_visibility_per_waypoint_list:
                if lidar_visibility_per_waypoint is not None and 'start_time' in lidar_visibility_per_waypoint:
                    lidar_visibility_per_waypoint_dict[lidar_visibility_per_waypoint['start_time']] = lidar_visibility_per_waypoint
                    waypoint_start_times.add(lidar_visibility_per_waypoint['start_time'])

        relative_localization_error_per_waypoint_dict = dict()
        relative_localization_error_per_waypoint_list = get_yaml_by_path(metrics_dict, ['relative_localization_error', 'relative_localization_error_per_waypoint_list'])
        if relative_localization_error_per_waypoint_list is not None:
            for relative_localization_error_per_waypoint in relative_localization_error_per_waypoint_list:
                if relative_localization_error_per_waypoint is not None and 'start_time' in relative_localization_error_per_waypoint:
                    relative_localization_error_per_waypoint_dict[relative_localization_error_per_waypoint['start_time']] = relative_localization_error_per_waypoint
                    waypoint_start_times.add(relative_localization_error_per_waypoint['start_time'])

        waypoint_relative_localization_error_per_waypoint_dict = dict()
        waypoint_relative_localization_error_per_waypoint_list = get_yaml_by_path(metrics_dict, ['waypoint_relative_localization_error', 'relative_localization_error_per_waypoint_list'])
        if waypoint_relative_localization_error_per_waypoint_list is not None:
            for waypoint_relative_localization_error_per_waypoint in waypoint_relative_localization_error_per_waypoint_list:
                if waypoint_relative_localization_error_per_waypoint is not None and 'start_time' in waypoint_relative_localization_error_per_waypoint:
                    waypoint_relative_localization_error_per_waypoint_dict[waypoint_relative_localization_error_per_waypoint['start_time']] = waypoint_relative_localization_error_per_waypoint
                    waypoint_start_times.add(waypoint_relative_localization_error_per_waypoint['start_time'])

        trajectory_length_per_waypoint_dict = dict()
        trajectory_length_per_waypoint_list = get_yaml_by_path(metrics_dict, ['trajectory_length_per_waypoint', 'trajectory_length_per_waypoint_list'])
        if trajectory_length_per_waypoint_list is not None:
            for trajectory_length_per_waypoint in trajectory_length_per_waypoint_list:
                if trajectory_length_per_waypoint is not None and 'start_time' in trajectory_length_per_waypoint:
                    trajectory_length_per_waypoint_dict[trajectory_length_per_waypoint['start_time']] = trajectory_length_per_waypoint
                    waypoint_start_times.add(trajectory_length_per_waypoint['start_time'])

        for waypoint_start_time in waypoint_start_times:
            run_record_per_waypoint = run_record.copy()

            run_record_per_waypoint['waypoint_start_time'] = waypoint_start_time

            run_record_per_waypoint['relative_localization_error_translation_mean'] = get_yaml_by_path(relative_localization_error_per_waypoint_dict, [waypoint_start_time, 'random_relations', 'translation', 'mean'])
            run_record_per_waypoint['relative_localization_error_rotation_mean'] = get_yaml_by_path(relative_localization_error_per_waypoint_dict, [waypoint_start_time, 'random_relations', 'rotation', 'mean'])

            all_geometric_similarity_metrics = get_yaml_by_path(geometric_similarity_per_waypoint_dict, [waypoint_start_time])
            if all_geometric_similarity_metrics is not None:
                for geometric_similarity_metric_name, geometric_similarity_metric_value in all_geometric_similarity_metrics.items():
                    run_record_per_waypoint['geometric_similarity_' + geometric_similarity_metric_name] = geometric_similarity_metric_value

            all_geometric_similarity_range_limit_metrics = get_yaml_by_path(geometric_similarity_range_limit_per_waypoint_dict, [waypoint_start_time])
            if all_geometric_similarity_range_limit_metrics is not None:
                for geometric_similarity_range_limit_metric_name, geometric_similarity_range_limit_metric_value in all_geometric_similarity_range_limit_metrics.items():
                    run_record_per_waypoint['geometric_similarity_range_limit_' + geometric_similarity_range_limit_metric_name] = geometric_similarity_range_limit_metric_value

            all_geometric_similarity_sensor_metrics = get_yaml_by_path(geometric_similarity_sensor_per_waypoint_dict, [waypoint_start_time])
            if all_geometric_similarity_sensor_metrics is not None:
                for geometric_similarity_sensor_metric_name, geometric_similarity_sensor_metric_value in all_geometric_similarity_sensor_metrics.items():
                    run_record_per_waypoint['geometric_similarity_sensor_' + geometric_similarity_sensor_metric_name] = geometric_similarity_sensor_metric_value

            all_lidar_visibility_metrics = get_yaml_by_path(lidar_visibility_per_waypoint_dict, [waypoint_start_time])
            if all_lidar_visibility_metrics is not None:
                for lidar_visibility_metric_name, lidar_visibility_metric_value in all_lidar_visibility_metrics.items():
                    run_record_per_waypoint['lidar_visibility_' + lidar_visibility_metric_name] = lidar_visibility_metric_value

            all_waypoint_relative_localization_metrics = get_yaml_by_path(waypoint_relative_localization_error_per_waypoint_dict, [waypoint_start_time])
            if all_waypoint_relative_localization_metrics is not None:
                for all_waypoint_relative_localization_metric_name,  all_waypoint_relative_localization_metric_value in all_waypoint_relative_localization_metrics.items():
                    run_record_per_waypoint['waypoint_relative_localization_error_' + all_waypoint_relative_localization_metric_name] = all_waypoint_relative_localization_metric_value

            all_trajectory_length_metrics = get_yaml_by_path(trajectory_length_per_waypoint_dict, [waypoint_start_time])
            if all_trajectory_length_metrics is not None:
                for all_trajectory_length_metric_name,  all_trajectory_length_metric_value in all_trajectory_length_metrics.items():
                    run_record_per_waypoint['trajectory_length_' + all_trajectory_length_metric_name] = all_trajectory_length_metric_value

            record_list.append(run_record_per_waypoint)

        print_info("collect_data: reading run data: {}% {}/{} {}".format(int((i + 1)*100/len(run_folders)), i, len(run_folders), path.basename(run_folder)), replace_previous_line=no_output)
        no_output = True

    df = pd.DataFrame(record_list)

    # save cache
    if cache_file_path is not None:
        cache = {'df': df, 'parameter_names': parameter_names}
        with open(cache_file_path, 'wb') as f:
            pickle.dump(cache, f, protocol=2)

    return df, parameter_names


if __name__ == '__main__':
    default_base_run_folder = "~/ds/performance_modelling/output/test_slam"

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the analysis of the benchmark results.')
    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder containing the result the runs. Defaults to {}'.format(default_base_run_folder),
                        type=str,
                        default=default_base_run_folder,
                        required=False)

    parser.add_argument('-i', dest='invalidate_cache',
                        help='If set, all the data is re-read.',
                        action='store_true',
                        default=True,  # TODO does nothing, repair or remove
                        required=False)

    args = parser.parse_args()
    import time
    s = time.time()
    run_data_df, params = collect_data(args.base_run_folder, args.invalidate_cache)
    pd.options.display.width = 200
    print(run_data_df)
    print(run_data_df.columns)
    print("collect_data: ", time.time() - s, "s")
