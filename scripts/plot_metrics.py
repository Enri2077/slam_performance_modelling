#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
import pickle
import sys
import pandas as pd
import pandas.core.groupby
from matplotlib.cm import get_cmap
from os import path

import matplotlib.pyplot as plt

from performance_modelling_py.utils import print_info, print_error


def cm_to_body_parts(*argv):
    inch = 2.54
    if isinstance(argv[0], tuple):
        return tuple(x_cm / inch for x_cm in argv[0])
    else:
        return tuple(x_cm / inch for x_cm in argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the analysis of the benchmark results.')

    # parser.add_argument('-r', dest='base_run_folder',
    #                     help='Folder containing the result the runs. Defaults to ~/ds/performance_modelling/output/test_slam/',
    #                     type=str,
    #                     default="~/ds/performance_modelling/output/test_slam/",
    #                     required=False)

    parser.add_argument('-i', dest='cache_file_path',
                        help='Folder containing the result the runs. Defaults to ~/ds/performance_modelling/output/test_slam/run_data_cache.pkl',
                        type=str,
                        default="~/ds/performance_modelling/output/test_slam/",
                        required=False)

    parser.add_argument('-o', dest='output_folder',
                        help='Folder in which the results will be placed. Defaults to ~/ds/performance_modelling/plots/test_slam/',
                        type=str,
                        default="~/ds/performance_modelling/plots/test_slam/",
                        required=False)

    parser.add_argument('--plot-everything', dest='plot_everything',
                        help='Plot everything.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-metrics-by-parameter', dest='plot_metrics_by_parameter',
                        help='Plot metrics by parameter using the specified aggregation function.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-trade-off', dest='plot_trade_off',
                        help='Plot trade-off between metrics by environment.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-metric-histograms', dest='plot_metric_histograms',
                        help='Plot metric histograms.',
                        action='store_true',
                        default=False,
                        required=False)

    args = parser.parse_args()
    output_folder = path.expanduser(args.output_folder)
    # base_run_folder = path.expanduser(args.base_run_folder)
    # cache_file_path = path.join(base_run_folder, "run_data_cache.pkl")
    cache_file_path = path.expanduser(args.cache_file_path)

    # if not path.isdir(base_run_folder):
    #     print_error("base_run_folder does not exists or is not a directory [{}]".format(base_run_folder))
    #     sys.exit(-1)

    if not path.exists(cache_file_path):
        print_error("cache_file_path does not exists [{}]".format(cache_file_path))
        sys.exit(-1)

    if not path.exists(output_folder):
        os.makedirs(output_folder)

    with open(cache_file_path, 'rb') as cache_file:
        cache = pickle.load(cache_file)

    df = cache['df']
    parameter_names = list(cache['parameter_names'])

    metric_optimisation_factor = {
        'max_memory': -1,
        'mean_absolute_error': -1,
        'normalised_cpu_time': -1,
        'run_execution_time': -1,
        'failure_rate': -1,
    }

    metric_names = set(metric_optimisation_factor.keys())

    # plot metrics in function of single configuration parameters
    if args.plot_everything or args.plot_metrics_by_parameter:
        aggregation_functions = {'var': pd.core.groupby.DataFrameGroupBy.var,
                                 'mean': pd.core.groupby.DataFrameGroupBy.mean,
                                 'median': pd.core.groupby.DataFrameGroupBy.median,
                                 'min': pd.core.groupby.DataFrameGroupBy.min,
                                 'max': pd.core.groupby.DataFrameGroupBy.max}

        for aggregation_function_name, aggregation_function in aggregation_functions.items():
            print_info("plot metrics by parameter using {a}".format(a=aggregation_function_name))
            metrics_by_parameter_folder = path.join(output_folder, "metrics_by_parameter_using_{a}".format(a=aggregation_function_name))
            if not path.exists(metrics_by_parameter_folder):
                os.makedirs(metrics_by_parameter_folder)

            for i, metric_name in enumerate(metric_names):

                for primary_parameter in parameter_names:
                    # plot lines for same-parameter metric values
                    fig, ax = plt.subplots()
                    fig.set_size_inches(*cm_to_body_parts(40, 40))
                    ax.margins(0.15)
                    ax.set_xlabel(primary_parameter)
                    ax.set_ylabel(metric_name)

                    secondary_parameters = list(set(parameter_names) - {primary_parameter})
                    for _, secondary_df in list(df.groupby(secondary_parameters, as_index=False)):
                        sorted_secondary_df = secondary_df.sort_values(by=primary_parameter)

                        environment_df = sorted_secondary_df.groupby(parameter_names, as_index=False)
                        aggregated_df = aggregation_function(environment_df)
                        print(aggregated_df)

                        ax.plot(aggregated_df[primary_parameter], aggregated_df[metric_name], marker='o', ms=3)

                    ax.grid(color='black', alpha=0.5, linestyle='solid')
                    fig.savefig(path.join(metrics_by_parameter_folder, "{}_by_{}_using_{}.svg".format(metric_name, primary_parameter, aggregation_function_name)), bbox_inches='tight')
                    plt.close(fig)

                print_info("plot metrics by parameter using {a}: {p}%".format(a=aggregation_function_name, p=(i + 1)*100/len(metric_names)), replace_previous_line=True)

    # plot trade-off between metrics
    if args.plot_everything or args.plot_trade_off:
        print_info("plot trade-off")

        trade_off_folder = path.join(output_folder, "trade_off")
        if not path.exists(trade_off_folder):
            os.makedirs(trade_off_folder)

        progress_counter = 0
        progress_total = len(metric_names)**2 - len(metric_names)

        for metric_x_name in metric_names:
            for metric_y_name in metric_names - {metric_x_name}:

                fig, ax = plt.subplots()
                fig.set_size_inches(*cm_to_body_parts(40, 40))
                ax.margins(0.15)
                ax.set_xlabel(metric_x_name)
                ax.set_ylabel(metric_y_name)

                x_s = metric_optimisation_factor[metric_x_name]
                y_s = metric_optimisation_factor[metric_y_name]

                df_grouped_by_environment = df.groupby('environment', as_index=False)
                color_map = get_cmap(lut=len(df_grouped_by_environment), name=None)
                for environment_index, (environment, environment_df) in enumerate(df_grouped_by_environment):

                    metric_x_values = list()
                    metric_y_values = list()
                    for metric_x_value, metric_y_value in environment_df.groupby(parameter_names, as_index=False).mean()[[metric_x_name, metric_y_name]].itertuples(index=False):
                        if pd.notnull(metric_x_value) and pd.notnull(metric_y_value):
                            metric_x_values.append(metric_x_value)
                            metric_y_values.append(metric_y_value)

                    ax.scatter(metric_x_values, metric_y_values, color=color_map(environment_index), alpha=0.5)

                    xy = zip(metric_x_values, metric_y_values)
                    dominant_points = list()
                    for x, y in xy:
                        dominated = False
                        for x_prime, y_prime in xy:
                            if x_s*x_prime >= x_s*x and y_s*y_prime >= y_s*y and (x_prime, y_prime) != (x, y):
                                dominated = True
                                break
                        if dominated:
                            continue
                        else:
                            dominant_points.append((x, y))

                    if len(dominant_points) > 0:
                        dominant_points = sorted(dominant_points, key=lambda p: p[0])
                        dominant_points_x, dominant_points_y = zip(*dominant_points)  # unzip
                    else:
                        dominant_points_x, dominant_points_y = max(xy, key=lambda d_x, d_y: (x_s * d_x, y_s * d_y))  # if no point is dominant, it means all points are aligned, so we can just take the maximum point (maximising both x and y)

                    ax.plot(dominant_points_x, dominant_points_y, marker='x', mew=3, ms=10, color=color_map(environment_index), label=environment)

                ax.legend()
                ax.grid(color='black', alpha=0.5, linestyle='solid')
                fig_name = "{x}_to_{y}.svg".format(y=metric_y_name, x=metric_x_name)
                fig.savefig(path.join(trade_off_folder, fig_name), bbox_inches='tight')
                plt.close(fig)

                progress_counter += 1
                print_info("plot trade-off: {}%".format(progress_counter * 100 / progress_total), replace_previous_line=True)

    # plot metrics histograms
    if args.plot_everything or args.plot_metric_histograms:
        print_info("plot metric histograms")

        metric_histograms_folder = path.join(output_folder, "metric_histograms")
        if not path.exists(metric_histograms_folder):
            os.makedirs(metric_histograms_folder)

        progress_counter = 0
        progress_total = len(metric_names)

        for metric_x_name in metric_names:
            fig, ax = plt.subplots()
            fig.set_size_inches(*cm_to_body_parts(40, 40))
            ax.margins(0.15)
            ax.set_xlabel(metric_x_name)

            metric_values = df[metric_x_name][pd.notnull(df[metric_x_name])]

            # plot histogram graph for metric x values
            ax.hist(metric_values, bins=100)

            fig_name = "{}.svg".format(metric_x_name)
            fig.savefig(path.join(metric_histograms_folder, fig_name), bbox_inches='tight')
            plt.close(fig)

            progress_counter += 1
            print_info("plot metric histograms: {}%".format(progress_counter * 100 / progress_total), replace_previous_line=True)

    # TODO plot metric sensitivity by param
