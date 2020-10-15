#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import shutil
import traceback

import rospy
import yaml
from xml.etree import ElementTree as et
import time
from os import path
import numpy as np

from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error
from performance_modelling_py.component_proxies.ros1_component import Component
from slam_performance_modelling_ros.metrics import compute_metrics


class BenchmarkRun(object):
    def __init__(self, run_id, run_output_folder, benchmark_log_path, environment_folder, parameters_combination_dict, benchmark_configuration_dict, show_ros_info, headless):

        # run configuration
        self.run_id = run_id
        self.run_output_folder = run_output_folder
        self.benchmark_log_path = benchmark_log_path
        self.run_parameters = parameters_combination_dict
        self.benchmark_configuration = benchmark_configuration_dict
        self.components_ros_output = 'screen' if show_ros_info else 'log'
        self.headless = headless

        # environment parameters
        self.environment_folder = environment_folder
        self.ground_truth_map_info_path = path.join(environment_folder, "data", "map.yaml")
        self.gazebo_model_path_env_var = ":".join(map(
            lambda p: path.expanduser(p),
            self.benchmark_configuration['gazebo_model_path_env_var'] + [path.dirname(path.abspath(self.environment_folder)), self.run_output_folder]
        ))
        self.gazebo_resource_path_env_var = ":".join(map(
            lambda p: path.expanduser(p),
            self.benchmark_configuration['gazebo_resource_path_env_var']
        ))
        self.gazebo_plugin_path_env_var = ":".join(map(
            lambda p: path.expanduser(p),
            self.benchmark_configuration['gazebo_plugin_path_env_var']
        ))
        self.slam_node = self.run_parameters['slam_node']
        beta_1, beta_2, beta_3, beta_4 = self.run_parameters['beta']
        laser_scan_max_range = self.run_parameters['laser_scan_max_range']
        laser_scan_fov_deg = self.run_parameters['laser_scan_fov_deg']
        laser_scan_fov_rad = (laser_scan_fov_deg-1)*np.pi/180
        map_resolution = self.run_parameters['map_resolution']
        ceres_loss_function = self.run_parameters['ceres_loss_function'] if self.slam_node == 'slam_toolbox' else None
        num_particles = self.run_parameters['particles'] if self.slam_node == 'gmapping' else None
        linear_update, angular_update = self.run_parameters['linear_angular_update']
        fewer_nav_goals = self.run_parameters['fewer_nav_goals']
        xy_goal_tolerance, yaw_goal_tolerance = self.run_parameters['goal_tolerance']

        random_traversal_path = self.benchmark_configuration['random_traversal_path']

        # run variables
        self.aborted = False

        # prepare folder structure
        run_configuration_path = path.join(self.run_output_folder, "components_configuration")
        run_info_file_path = path.join(self.run_output_folder, "run_info.yaml")
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        os.mkdir(run_configuration_path)

        # components original configuration paths
        components_configurations_folder = path.expanduser(self.benchmark_configuration['components_configurations_folder'])
        original_supervisor_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['supervisor'])
        original_gmapping_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['gmapping'])
        original_slam_toolbox_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['slam_toolbox'])
        original_move_base_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['move_base'])
        original_gazebo_world_model_path = path.join(environment_folder, "gazebo", "gazebo_environment.model")
        original_gazebo_robot_model_config_path = path.join(environment_folder, "gazebo", "robot", "model.config")
        original_gazebo_robot_model_sdf_path = path.join(environment_folder, "gazebo", "robot", "model.sdf")
        original_robot_urdf_path = path.join(environment_folder, "gazebo", "robot.urdf")

        # components configuration relative paths
        supervisor_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['supervisor'])
        gmapping_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['gmapping'])
        slam_toolbox_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['slam_toolbox'])
        move_base_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['move_base'])
        gazebo_world_model_relative_path = path.join("components_configuration", "gazebo", "gazebo_environment.model")
        gazebo_robot_model_config_relative_path = path.join("components_configuration", "gazebo", "robot", "model.config")
        gazebo_robot_model_sdf_relative_path = path.join("components_configuration", "gazebo", "robot", "model.sdf")
        robot_gt_urdf_relative_path = path.join("components_configuration", "gazebo", "robot_gt.urdf")
        robot_realistic_urdf_relative_path = path.join("components_configuration", "gazebo", "robot_realistic.urdf")

        # components configuration paths in run folder
        self.supervisor_configuration_path = path.join(self.run_output_folder, supervisor_configuration_relative_path)
        self.gmapping_configuration_path = path.join(self.run_output_folder, gmapping_configuration_relative_path)
        self.slam_toolbox_configuration_path = path.join(self.run_output_folder, slam_toolbox_configuration_relative_path)
        self.move_base_configuration_path = path.join(self.run_output_folder, move_base_configuration_relative_path)
        self.gazebo_world_model_path = path.join(self.run_output_folder, gazebo_world_model_relative_path)
        gazebo_robot_model_config_path = path.join(self.run_output_folder, gazebo_robot_model_config_relative_path)
        gazebo_robot_model_sdf_path = path.join(self.run_output_folder, gazebo_robot_model_sdf_relative_path)
        self.robot_gt_urdf_path = path.join(self.run_output_folder, robot_gt_urdf_relative_path)
        self.robot_realistic_urdf_path = path.join(self.run_output_folder, robot_realistic_urdf_relative_path)

        # copy the configuration of the supervisor to the run folder and update its parameters
        with open(original_supervisor_configuration_path) as supervisor_configuration_file:
            supervisor_configuration = yaml.safe_load(supervisor_configuration_file)
        supervisor_configuration['run_output_folder'] = self.run_output_folder
        supervisor_configuration['pid_father'] = os.getpid()
        supervisor_configuration['ground_truth_map_info_path'] = self.ground_truth_map_info_path
        supervisor_configuration['fewer_nav_goals'] = fewer_nav_goals
        supervisor_configuration['random_traversal_path'] = random_traversal_path
        if not path.exists(path.dirname(self.supervisor_configuration_path)):
            os.makedirs(path.dirname(self.supervisor_configuration_path))
        with open(self.supervisor_configuration_path, 'w') as supervisor_configuration_file:
            yaml.dump(supervisor_configuration, supervisor_configuration_file, default_flow_style=False)

        if self.slam_node == 'gmapping':
            # copy the configuration of gmapping to the run folder and update its parameters
            with open(original_gmapping_configuration_path) as original_gmapping_configuration_file:
                gmapping_configuration = yaml.safe_load(original_gmapping_configuration_file)
            gmapping_configuration['stt'] = max(0.005, beta_1)  # rad/rad, Odometry error in rotation as a function of rotation (theta/theta)
            gmapping_configuration['str'] = max(0.005, beta_2)  # rad/m, Odometry error in rotation as a function of translation (theta/rho)
            gmapping_configuration['srr'] = max(0.005, beta_3)  # m/m, Odometry error in translation as a function of translation (rho/rho)
            gmapping_configuration['srt'] = max(0.005, beta_4)  # m/rad, Odometry error in translation as a function of rotation (rho/theta)
            gmapping_configuration['num_particles'] = num_particles
            gmapping_configuration['linearUpdate'] = linear_update
            gmapping_configuration['angularUpdate'] = angular_update
            gmapping_configuration['delta'] = map_resolution
            gmapping_configuration['maxUrange'] = laser_scan_max_range
            if not path.exists(path.dirname(self.gmapping_configuration_path)):
                os.makedirs(path.dirname(self.gmapping_configuration_path))
            with open(self.gmapping_configuration_path, 'w') as gmapping_configuration_file:
                yaml.dump(gmapping_configuration, gmapping_configuration_file, default_flow_style=False)

        elif self.slam_node == 'slam_toolbox':
            # copy the configuration of slam_toolbox to the run folder and update its parameters
            with open(original_slam_toolbox_configuration_path) as original_slam_toolbox_configuration_file:
                slam_toolbox_configuration = yaml.safe_load(original_slam_toolbox_configuration_file)
            slam_toolbox_configuration['ceres_loss_function'] = ceres_loss_function
            slam_toolbox_configuration['minimum_travel_distance'] = linear_update
            slam_toolbox_configuration['minimum_travel_heading'] = angular_update
            slam_toolbox_configuration['max_laser_range'] = laser_scan_max_range
            slam_toolbox_configuration['resolution'] = map_resolution
            if not path.exists(path.dirname(self.slam_toolbox_configuration_path)):
                os.makedirs(path.dirname(self.slam_toolbox_configuration_path))
            with open(self.slam_toolbox_configuration_path, 'w') as slam_toolbox_configuration_file:
                yaml.dump(slam_toolbox_configuration, slam_toolbox_configuration_file, default_flow_style=False)

        else:
            raise ValueError()

        # copy the configuration of move_base to the run folder
        with open(original_move_base_configuration_path) as original_move_base_configuration_file:
            move_base_configuration = yaml.safe_load(original_move_base_configuration_file)
        move_base_configuration['DWAPlannerROS']['xy_goal_tolerance'] = xy_goal_tolerance
        move_base_configuration['DWAPlannerROS']['yaw_goal_tolerance'] = yaw_goal_tolerance
        if not path.exists(path.dirname(self.move_base_configuration_path)):
            os.makedirs(path.dirname(self.move_base_configuration_path))
        with open(self.move_base_configuration_path, 'w') as move_base_configuration_file:
            yaml.dump(move_base_configuration, move_base_configuration_file, default_flow_style=False)

        # copy the configuration of the gazebo world model to the run folder and update its parameters
        gazebo_original_world_model_tree = et.parse(original_gazebo_world_model_path)
        gazebo_original_world_model_root = gazebo_original_world_model_tree.getroot()
        gazebo_original_world_model_root.findall(".//include[@include_id='robot_model']/uri")[0].text = path.join("model://", path.dirname(gazebo_robot_model_sdf_relative_path))
        if not path.exists(path.dirname(self.gazebo_world_model_path)):
            os.makedirs(path.dirname(self.gazebo_world_model_path))
        gazebo_original_world_model_tree.write(self.gazebo_world_model_path)

        # copy the configuration of the gazebo robot sdf model to the run folder and update its parameters
        gazebo_robot_model_sdf_tree = et.parse(original_gazebo_robot_model_sdf_path)
        gazebo_robot_model_sdf_root = gazebo_robot_model_sdf_tree.getroot()
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/scan/horizontal/samples")[0].text = str(int(laser_scan_fov_deg))
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/scan/horizontal/min_angle")[0].text = str(float(-laser_scan_fov_rad/2))
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/scan/horizontal/max_angle")[0].text = str(float(+laser_scan_fov_rad/2))
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/range/max")[0].text = str(float(laser_scan_max_range))
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/plugin[@name='turtlebot3_laserscan_realistic']/frameName")[0].text = "base_scan_realistic"
        if beta_1 == 0.0 and beta_2 == 0.0 and beta_3 == 0.0 and beta_4 == 0.0:
            gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/odometrySource")[0].text = "world"
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha1")[0].text = str(beta_1)
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha2")[0].text = str(beta_2)
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha3")[0].text = str(beta_3)
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha4")[0].text = str(beta_4)
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/odometryTopic")[0].text = "odom"
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/odometryFrame")[0].text = "odom_realistic"
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/robotBaseFrame")[0].text = "base_footprint_realistic"
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/groundTruthParentFrame")[0].text = "map"
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/groundTruthRobotBaseFrame")[0].text = "base_footprint_gt"
        if not path.exists(path.dirname(gazebo_robot_model_sdf_path)):
            os.makedirs(path.dirname(gazebo_robot_model_sdf_path))
        gazebo_robot_model_sdf_tree.write(gazebo_robot_model_sdf_path)

        # copy the configuration of the gazebo robot model to the run folder
        if not path.exists(path.dirname(gazebo_robot_model_config_path)):
            os.makedirs(path.dirname(gazebo_robot_model_config_path))
        shutil.copyfile(original_gazebo_robot_model_config_path, gazebo_robot_model_config_path)

        # copy the configuration of the robot urdf to the run folder and update the link names for ground truth data
        robot_gt_urdf_tree = et.parse(original_robot_urdf_path)
        robot_gt_urdf_root = robot_gt_urdf_tree.getroot()
        for link_element in robot_gt_urdf_root.findall(".//link"):
            link_element.attrib['name'] = "{}_gt".format(link_element.attrib['name'])
        for joint_link_element in robot_gt_urdf_root.findall(".//*[@link]"):
            joint_link_element.attrib['link'] = "{}_gt".format(joint_link_element.attrib['link'])
        if not path.exists(path.dirname(self.robot_gt_urdf_path)):
            os.makedirs(path.dirname(self.robot_gt_urdf_path))
        robot_gt_urdf_tree.write(self.robot_gt_urdf_path)

        # copy the configuration of the robot urdf to the run folder and update the link names for realistic data
        robot_realistic_urdf_tree = et.parse(original_robot_urdf_path)
        robot_realistic_urdf_root = robot_realistic_urdf_tree.getroot()
        for link_element in robot_realistic_urdf_root.findall(".//link"):
            link_element.attrib['name'] = "{}_realistic".format(link_element.attrib['name'])
        for joint_link_element in robot_realistic_urdf_root.findall(".//*[@link]"):
            joint_link_element.attrib['link'] = "{}_realistic".format(joint_link_element.attrib['link'])
        if not path.exists(path.dirname(self.robot_realistic_urdf_path)):
            os.makedirs(path.dirname(self.robot_realistic_urdf_path))
        robot_realistic_urdf_tree.write(self.robot_realistic_urdf_path)

        # write run info to file
        run_info_dict = dict()
        run_info_dict['run_id'] = self.run_id
        run_info_dict['run_folder'] = self.run_output_folder
        run_info_dict['environment_folder'] = environment_folder
        run_info_dict['run_parameters'] = self.run_parameters
        run_info_dict['local_components_configuration'] = {
            'supervisor': supervisor_configuration_relative_path,
            'move_base': move_base_configuration_relative_path,
            'gazebo_world_model': gazebo_world_model_relative_path,
            'gazebo_robot_model_sdf': gazebo_robot_model_sdf_relative_path,
            'gazebo_robot_model_config': gazebo_robot_model_config_relative_path,
            'robot_gt_urdf': robot_gt_urdf_relative_path,
            'robot_realistic_urdf': robot_realistic_urdf_relative_path,
        }

        if self.slam_node == 'gmapping':
            run_info_dict['local_components_configuration']['gmapping'] = gmapping_configuration_relative_path
        elif self.slam_node == 'slam_toolbox':
            run_info_dict['local_components_configuration']['slam_toolbox'] = slam_toolbox_configuration_relative_path
        else:
            raise ValueError()

        with open(run_info_file_path, 'w') as run_info_file:
            yaml.dump(run_info_dict, run_info_file, default_flow_style=False)

    def log(self, event):

        if not path.exists(self.benchmark_log_path):
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("timestamp, run_id, event\n")

        t = time.time()

        print_info("t: {t}, run: {run_id}, event: {event}".format(t=t, run_id=self.run_id, event=event))
        try:
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("{t}, {run_id}, {event}\n".format(t=t, run_id=self.run_id, event=event))
        except IOError as e:
            print_error("benchmark_log: could not write event to file: {t}, {run_id}, {event}".format(t=t, run_id=self.run_id, event=event))
            print_error(e)

    def execute_run(self):

        # components parameters
        rviz_params = {
            'headless': self.headless,
        }
        ground_truth_map_server_params = {
            'map': self.ground_truth_map_info_path,
        }
        environment_params = {
            'world_model_file': self.gazebo_world_model_path,
            'robot_gt_urdf_file': self.robot_gt_urdf_path,
            'robot_realistic_urdf_file': self.robot_realistic_urdf_path,
            'headless': True,
        }
        if self.slam_node == 'gmapping':
            slam_params = {
                'params_file': self.gmapping_configuration_path,
            }
        elif self.slam_node == 'slam_toolbox':
            slam_params = {
                'params_file': self.slam_toolbox_configuration_path,
            }
        else:
            raise ValueError()
        navigation_params = {
            'params_file': self.move_base_configuration_path,
        }
        supervisor_params = {
            'params_file': self.supervisor_configuration_path,
        }
        recorder_benchmark_data_params = {
            'bag_file_path': path.join(self.run_output_folder, "benchmark_data.bag"),
            'topics': "/base_footprint_gt /cmd_vel /initialpose /map_gt /map_gt_metadata /map_gt_updates /map /map_metadata /map_updates /odom /particlecloud /gmapping/entropy /rosout /rosout_agg /scan /scan_gt /tf /tf_static /traversal_path",
        }

        # declare components
        roscore = Component('roscore', 'slam_performance_modelling', 'roscore.launch')
        environment = Component('gazebo', 'slam_performance_modelling', 'gazebo.launch', environment_params)
        rviz = Component('rviz', 'slam_performance_modelling', 'rviz.launch', rviz_params)
        recorder_benchmark_data = Component('recorder_sensor_data', 'slam_performance_modelling', 'rosbag_recorder.launch', recorder_benchmark_data_params)
        if self.slam_node == 'gmapping':
            slam = Component('gmapping', 'slam_performance_modelling', 'gmapping.launch', slam_params)
        elif self.slam_node == 'slam_toolbox':
            slam = Component('slam_toolbox', 'slam_performance_modelling', 'slam_toolbox_online_async.launch', slam_params)
        else:
            raise ValueError()
        ground_truth_map_server = Component('ground_truth_map_server', 'slam_performance_modelling', 'ground_truth_map_server.launch', ground_truth_map_server_params)
        navigation = Component('move_base', 'slam_performance_modelling', 'move_base.launch', navigation_params)
        supervisor = Component('supervisor', 'slam_performance_modelling', 'slam_benchmark_supervisor.launch', supervisor_params)

        # set gazebo's environment variables
        os.environ['GAZEBO_MODEL_PATH'] = self.gazebo_model_path_env_var
        os.environ['GAZEBO_RESOURCE_PATH'] = self.gazebo_resource_path_env_var
        os.environ['GAZEBO_PLUGIN_PATH'] = self.gazebo_plugin_path_env_var

        # launch roscore and setup a node to monitor ros
        roscore.launch()
        rospy.init_node("benchmark_monitor", anonymous=True)

        # launch components
        environment.launch()
        rviz.launch()
        recorder_benchmark_data.launch()
        slam.launch()
        ground_truth_map_server.launch()
        navigation.launch()
        supervisor.launch()

        # launch components and wait for the supervisor to finish
        self.log(event="waiting_supervisor_finish")
        supervisor.wait_to_finish()
        self.log(event="supervisor_shutdown")

        # check if the rosnode is still ok, otherwise the ros infrastructure has been shutdown and the benchmark is aborted
        if rospy.is_shutdown():
            print_error("execute_run: supervisor finished by ros_shutdown")
            self.aborted = True

        # shut down components
        ground_truth_map_server.shutdown()
        navigation.shutdown()
        slam.shutdown()
        recorder_benchmark_data.shutdown()
        rviz.shutdown()
        environment.shutdown()
        roscore.shutdown()
        print_info("execute_run: components shutdown completed")

        # compute all relevant metrics and visualisations
        # noinspection PyBroadException
        try:
            self.log(event="start_compute_metrics")
            compute_metrics(self.run_output_folder)
        except:
            print_error("failed metrics computation")
            print_error(traceback.format_exc())

        self.log(event="run_end")
        print_info("run {run_id} completed".format(run_id=self.run_id))
