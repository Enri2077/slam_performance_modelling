#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import shutil

import rospy
import yaml
from xml.etree import ElementTree as et
import time
from os import path
import numpy as np

from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error
from performance_modelling_py.component_proxies.ros1_component import Component
from performance_modelling_py.benchmark_execution.log_software_versions import log_packages_and_repos


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
        self.environment_type = self.benchmark_configuration['environment_type'] if 'environment_type' in self.benchmark_configuration else 'simulation'
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
        laser_scan_max_range = self.run_parameters['laser_scan_max_range']
        laser_scan_fov_deg = self.run_parameters['laser_scan_fov_deg']
        laser_scan_fov_rad = (laser_scan_fov_deg-1)*np.pi/180
        map_resolution = self.run_parameters['map_resolution']
        ceres_loss_function = self.run_parameters['ceres_loss_function'] if self.slam_node == 'slam_toolbox' else None
        num_particles = self.run_parameters['particles'] if self.slam_node == 'gmapping' else None
        linear_update, angular_update = self.run_parameters['linear_angular_update']

        if self.environment_type == 'dataset':
            random_traversal_path = False
            fewer_nav_goals = False
            beta_1 = beta_2 = beta_3 = beta_4 = None
            xy_goal_tolerance = yaw_goal_tolerance = None
            dataset_odom = self.run_parameters['dataset_odom']
            self.real_data_rosbag_path = path.join(self.environment_folder, "data", "recorded_data", "odom-{odom}_laser_scan_fov_deg-{laser_scan_fov_deg}_laser_scan_max_range-{laser_scan_max_range}.bag".format(
                odom=dataset_odom,
                laser_scan_fov_deg=float(laser_scan_fov_deg),
                laser_scan_max_range=float(laser_scan_max_range)
            ))
        elif self.environment_type == 'simulation':
            random_traversal_path = self.benchmark_configuration['random_traversal_path']
            fewer_nav_goals = self.run_parameters['fewer_nav_goals']
            beta_1, beta_2, beta_3, beta_4 = self.run_parameters['beta']
            xy_goal_tolerance, yaw_goal_tolerance = self.run_parameters['goal_tolerance']
            self.real_data_rosbag_path = None
        else:
            raise ValueError()

        # run variables
        self.aborted = False

        # prepare folder structure
        run_configuration_path = path.join(self.run_output_folder, "components_configuration")
        run_info_file_path = path.join(self.run_output_folder, "run_info.yaml")
        source_workspace_path = self.benchmark_configuration['source_workspace_path']
        software_versions_log_path = path.join(self.run_output_folder, "software_versions_log")
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        os.mkdir(run_configuration_path)

        # components original configuration paths
        components_configurations_folder = path.expanduser(self.benchmark_configuration['components_configurations_folder'])
        original_supervisor_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['supervisor'])
        original_gmapping_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['gmapping'])
        original_slam_toolbox_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['slam_toolbox'])
        original_hector_slam_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['hector_slam'])
        original_move_base_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['move_base']) if self.environment_type == 'simulation' else None
        original_gazebo_world_model_path = path.join(environment_folder, "gazebo", "gazebo_environment.model") if self.environment_type == 'simulation' else None
        original_gazebo_robot_model_config_path = path.join(environment_folder, "gazebo", "robot", "model.config") if self.environment_type == 'simulation' else None
        original_gazebo_robot_model_sdf_path = path.join(environment_folder, "gazebo", "robot", "model.sdf") if self.environment_type == 'simulation' else None
        original_robot_urdf_path = path.join(environment_folder, "gazebo", "robot.urdf") if self.environment_type == 'simulation' else None

        # components configuration relative paths
        supervisor_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['supervisor'])
        gmapping_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['gmapping'])
        slam_toolbox_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['slam_toolbox'])
        hector_slam_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['hector_slam'])
        move_base_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['move_base']) if self.environment_type == 'simulation' else None
        gazebo_world_model_relative_path = path.join("components_configuration", "gazebo", "gazebo_environment.model") if self.environment_type == 'simulation' else None
        gazebo_robot_model_config_relative_path = path.join("components_configuration", "gazebo", "robot", "model.config") if self.environment_type == 'simulation' else None
        gazebo_robot_model_sdf_relative_path = path.join("components_configuration", "gazebo", "robot", "model.sdf") if self.environment_type == 'simulation' else None
        robot_gt_urdf_relative_path = path.join("components_configuration", "gazebo", "robot_gt.urdf") if self.environment_type == 'simulation' else None
        robot_realistic_urdf_relative_path = path.join("components_configuration", "gazebo", "robot_realistic.urdf") if self.environment_type == 'simulation' else None

        # components configuration paths in run folder
        self.supervisor_configuration_path = path.join(self.run_output_folder, supervisor_configuration_relative_path)
        self.gmapping_configuration_path = path.join(self.run_output_folder, gmapping_configuration_relative_path)
        self.slam_toolbox_configuration_path = path.join(self.run_output_folder, slam_toolbox_configuration_relative_path)
        self.hector_slam_configuration_path = path.join(self.run_output_folder, hector_slam_configuration_relative_path)
        self.move_base_configuration_path = path.join(self.run_output_folder, move_base_configuration_relative_path) if self.environment_type == 'simulation' else None
        self.gazebo_world_model_path = path.join(self.run_output_folder, gazebo_world_model_relative_path) if self.environment_type == 'simulation' else None
        gazebo_robot_model_config_path = path.join(self.run_output_folder, gazebo_robot_model_config_relative_path) if self.environment_type == 'simulation' else None
        gazebo_robot_model_sdf_path = path.join(self.run_output_folder, gazebo_robot_model_sdf_relative_path) if self.environment_type == 'simulation' else None
        self.robot_gt_urdf_path = path.join(self.run_output_folder, robot_gt_urdf_relative_path) if self.environment_type == 'simulation' else None
        self.robot_realistic_urdf_path = path.join(self.run_output_folder, robot_realistic_urdf_relative_path) if self.environment_type == 'simulation' else None

        # copy the configuration of the supervisor to the run folder and update its parameters
        with open(original_supervisor_configuration_path) as supervisor_configuration_file:
            supervisor_configuration = yaml.safe_load(supervisor_configuration_file)
        supervisor_configuration['run_output_folder'] = self.run_output_folder
        supervisor_configuration['pid_father'] = os.getpid()
        supervisor_configuration['ground_truth_map_info_path'] = self.ground_truth_map_info_path
        supervisor_configuration['fewer_nav_goals'] = fewer_nav_goals
        supervisor_configuration['random_traversal_path'] = random_traversal_path
        supervisor_configuration['environment_type'] = self.environment_type
        supervisor_configuration['robot_base_frame'] = "base_link_realistic" if self.environment_type == 'simulation' else "base_link"
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
            gmapping_configuration['odom_frame'] = "odom_realistic" if self.environment_type == 'simulation' else "odom"
            gmapping_configuration['base_frame'] = "base_footprint_realistic" if self.environment_type == 'simulation' else "base_link"
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
            slam_toolbox_configuration['odom_frame'] = "odom_realistic" if self.environment_type == 'simulation' else "odom"
            slam_toolbox_configuration['base_frame'] = "base_footprint_realistic" if self.environment_type == 'simulation' else "base_link"
            slam_toolbox_configuration['ceres_loss_function'] = ceres_loss_function
            slam_toolbox_configuration['minimum_travel_distance'] = linear_update
            slam_toolbox_configuration['minimum_travel_heading'] = angular_update
            slam_toolbox_configuration['max_laser_range'] = laser_scan_max_range
            slam_toolbox_configuration['resolution'] = map_resolution
            if not path.exists(path.dirname(self.slam_toolbox_configuration_path)):
                os.makedirs(path.dirname(self.slam_toolbox_configuration_path))
            with open(self.slam_toolbox_configuration_path, 'w') as slam_toolbox_configuration_file:
                yaml.dump(slam_toolbox_configuration, slam_toolbox_configuration_file, default_flow_style=False)

        elif self.slam_node == 'hector_slam':
            # copy the configuration of hector_slam to the run folder and update its parameters
            with open(original_hector_slam_configuration_path) as original_hector_slam_configuration_file:
                hector_slam_configuration = yaml.safe_load(original_hector_slam_configuration_file)
            hector_slam_configuration['odom_frame'] = "odom_realistic" if self.environment_type == 'simulation' else "odom"
            hector_slam_configuration['base_frame'] = "base_footprint_realistic" if self.environment_type == 'simulation' else "base_link"
            hector_slam_configuration['map_update_distance_thresh'] = linear_update
            hector_slam_configuration['map_update_angle_thresh'] = angular_update
            if not path.exists(path.dirname(self.hector_slam_configuration_path)):
                os.makedirs(path.dirname(self.hector_slam_configuration_path))
            with open(self.hector_slam_configuration_path, 'w') as hector_slam_configuration_file:
                yaml.dump(hector_slam_configuration, hector_slam_configuration_file, default_flow_style=False)

        else:
            raise ValueError()

        if self.environment_type == 'simulation':
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
        run_info_dict['benchmark_configuration'] = self.benchmark_configuration
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
        elif self.slam_node == 'hector_slam':
            run_info_dict['local_components_configuration']['hector_slam'] = hector_slam_configuration_relative_path
        else:
            raise ValueError()

        with open(run_info_file_path, 'w') as run_info_file:
            yaml.dump(run_info_dict, run_info_file, default_flow_style=False)

        # log packages and software versions and status
        log_packages_and_repos(source_workspace_path=source_workspace_path, log_dir_path=software_versions_log_path)

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

        if self.environment_type == 'dataset':
            navigation_params = None
            simulated_environment_params = None
            recorded_environment_params = {
                'rosbag_file_path': self.real_data_rosbag_path,
            }
        elif self.environment_type == 'simulation':
            recorded_environment_params = None
            simulated_environment_params = {
                'world_model_file': self.gazebo_world_model_path,
                'robot_gt_urdf_file': self.robot_gt_urdf_path,
                'robot_realistic_urdf_file': self.robot_realistic_urdf_path,
                'headless': True,
            }
            navigation_params = {
                'params_file': self.move_base_configuration_path,
            }
        else:
            raise ValueError()

        if self.slam_node == 'gmapping':
            slam_params = {
                'params_file': self.gmapping_configuration_path,
            }
        elif self.slam_node == 'slam_toolbox':
            slam_params = {
                'params_file': self.slam_toolbox_configuration_path,
            }
        elif self.slam_node == 'hector_slam':
            slam_params = {
                'params_file': self.hector_slam_configuration_path,
            }
        else:
            raise ValueError()

        supervisor_params = {
            'params_file': self.supervisor_configuration_path,
        }
        recorder_benchmark_data_params = {
            'bag_file_path': path.join(self.run_output_folder, "benchmark_data.bag"),
            'topics': "/base_footprint_gt /cmd_vel /initialpose /map_gt /map_gt_metadata /map_gt_updates /map /map_metadata /map_updates /odom /particlecloud /gmapping/entropy /rosout /rosout_agg /scan /scan_gt /tf /tf_static /traversal_path",
        }

        # declare components
        roscore = Component('roscore', 'slam_performance_modelling', 'roscore.launch')

        if self.environment_type == 'dataset':
            environment = Component('rosbag_player', 'slam_performance_modelling', 'rosbag_player.launch', recorded_environment_params)
            navigation = None
        elif self.environment_type == 'simulation':
            environment = Component('gazebo', 'slam_performance_modelling', 'gazebo.launch', simulated_environment_params)
            navigation = Component('move_base', 'slam_performance_modelling', 'move_base.launch', navigation_params)
        else:
            raise ValueError()

        rviz = Component('rviz', 'slam_performance_modelling', 'rviz.launch', rviz_params)
        recorder_benchmark_data = Component('recorder_sensor_data', 'slam_performance_modelling', 'rosbag_recorder.launch', recorder_benchmark_data_params)
        if self.slam_node == 'gmapping':
            slam = Component('gmapping', 'slam_performance_modelling', 'gmapping.launch', slam_params)
        elif self.slam_node == 'slam_toolbox':
            slam = Component('slam_toolbox', 'slam_performance_modelling', 'slam_toolbox_online_async.launch', slam_params)
        elif self.slam_node == 'hector_slam':
            slam = Component('hector_slam', 'slam_performance_modelling', 'hector_slam.launch', slam_params)
        else:
            raise ValueError()
        ground_truth_map_server = Component('ground_truth_map_server', 'slam_performance_modelling', 'ground_truth_map_server.launch', ground_truth_map_server_params)
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
        if self.environment_type == 'simulation':
            navigation.launch()
        supervisor.launch()

        # launch components and wait for the supervisor to finish
        if self.environment_type == 'dataset':
            self.log(event="waiting_recorded_dataset_finish")
            environment.wait_to_finish()
            self.log(event="recorded_dataset_shutdown")
        elif self.environment_type == 'simulation':
            self.log(event="waiting_supervisor_finish")
            supervisor.wait_to_finish()
            self.log(event="supervisor_shutdown")
        else:
            raise ValueError()

        # check if the rosnode is still ok, otherwise the ros infrastructure has been shutdown and the benchmark is aborted
        if rospy.is_shutdown():
            print_error("execute_run: supervisor finished by ros_shutdown")
            self.aborted = True

        # shut down components
        if self.environment_type == 'dataset':
            supervisor.shutdown()
        elif self.environment_type == 'simulation':
            navigation.shutdown()
            environment.shutdown()
        else:
            raise ValueError()
        ground_truth_map_server.shutdown()
        slam.shutdown()
        recorder_benchmark_data.shutdown()
        rviz.shutdown()
        roscore.shutdown()
        print_info("execute_run: components shutdown completed")

        # compute all relevant metrics and visualisations
        # noinspection PyBroadException
        # try:
        #     self.log(event="start_compute_metrics")
        #     compute_metrics(self.run_output_folder)
        # except KeyboardInterrupt:
        #     print_info("\nmetrics computation interrupted")
        # except:
        #     print_error("failed metrics computation")
        #     print_error(traceback.format_exc())

        self.log(event="run_end")
        print_info("run {run_id} completed".format(run_id=self.run_id))
