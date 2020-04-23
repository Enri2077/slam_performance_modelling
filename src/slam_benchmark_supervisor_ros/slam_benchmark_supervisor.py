#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import pickle
import psutil

import rospy
import tf2_ros
import tf
import tf.transformations
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from image_utils import save_map_image, map_changed
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, Bool

import os
from os import path
import numpy as np

from performance_modelling_py.utils import backup_file_if_exists, print_info


class SlamBenchmarkSupervisor:
    def __init__(self):
        # general parameters
        exploration_finished_topic = rospy.get_param('~exploration_finished_topic')
        base_link_ground_truth_topic = rospy.get_param('~base_link_ground_truth_topic')
        gmapping_correction_topic = rospy.get_param('~gmapping_correction_topic')
        cmd_vel_topic = rospy.get_param('~cmd_vel_topic')
        map_topic = rospy.get_param('~map_topic')
        scan_topic = rospy.get_param('~scan_topic')

        # run parameters
        self.run_timeout = rospy.get_param('~run_timeout')
        self.write_base_link_poses_period = rospy.get_param('~write_base_link_poses_period')
        self.map_snapshot_period = rospy.get_param('~map_snapshot_period')
        self.ps_snapshot_period = rospy.get_param('~ps_snapshot_period')
        self.map_steady_state_period = rospy.get_param('~map_steady_state_period')
        self.run_output_folder = rospy.get_param('~run_output_folder')
        self.ps_pid_father = rospy.get_param('~pid_father')
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.map_output_folder = path.join(self.benchmark_data_folder, "map_snapshots")
        self.ps_output_folder = path.join(self.benchmark_data_folder, "ps_snapshots")

        self.map_change_threshold = rospy.get_param('~map_change_threshold')
        self.map_size_change_threshold = rospy.get_param('~map_size_change_threshold')  # percentage of increased area
        self.map_occupied_threshold = rospy.get_param('~map_occupied_threshold')
        self.map_free_threshold = rospy.get_param('~map_free_threshold')

        # run variables
        self.terminate = False
        self.initial_ground_truth_pose = None
        self.map_snapshot_count = 0
        self.ps_snapshot_count = 0
        self.last_map_msg = None
        self.ps_processes = psutil.Process(self.ps_pid_father).children(recursive=True)  # list of processes children of the benchmark script, i.e., all ros nodes of the benchmark including this one

        # prepare folder structure
        if not path.exists(self.benchmark_data_folder):
            os.makedirs(self.benchmark_data_folder)

        if not path.exists(self.map_output_folder):
            os.makedirs(self.map_output_folder)

        if not path.exists(self.ps_output_folder):
            os.makedirs(self.ps_output_folder)

        # file paths for benchmark data
        self.base_link_correction_poses_file_path = path.join(self.benchmark_data_folder, "base_link_correction_poses")
        self.base_link_poses_file_path = path.join(self.benchmark_data_folder, "base_link_poses")
        backup_file_if_exists(self.base_link_poses_file_path)

        self.ground_truth_poses_file_path = path.join(self.benchmark_data_folder, "ground_truth_poses")
        backup_file_if_exists(self.ground_truth_poses_file_path)

        self.cmd_vel_twists_file_path = path.join(self.benchmark_data_folder, "cmd_vel_twists")
        backup_file_if_exists(self.cmd_vel_twists_file_path)

        self.scans_file_path = path.join(self.benchmark_data_folder, "scans")
        backup_file_if_exists(self.scans_file_path)

        self.run_events_file_path = path.join(self.benchmark_data_folder, "run_events.csv")
        self.init_run_events_file()

        # setup timers, buffers and subscribers
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.run_timeout_timer = rospy.Timer(rospy.Duration.from_sec(self.run_timeout), self.run_timeout_callback, oneshot=True)
        self.write_base_link_poses_timer = rospy.Timer(rospy.Duration.from_sec(self.write_base_link_poses_period), self.write_base_link_poses_timer_callback)
        self.save_map_snapshot_timer = rospy.Timer(rospy.Duration.from_sec(self.map_snapshot_period), self.save_map_snapshot_timer_callback)
        self.ps_snapshot_timer = rospy.Timer(rospy.Duration.from_sec(self.ps_snapshot_period), self.ps_snapshot_timer_callback)
        self.explorer_finished_subscriber = rospy.Subscriber(exploration_finished_topic, Bool, self.exploration_finished_callback, queue_size=10)
        self.ground_truth_pose_subscriber = rospy.Subscriber(base_link_ground_truth_topic, Odometry, self.ground_truth_pose_callback, queue_size=10)
        self.gmapping_correction_subscriber = rospy.Subscriber(gmapping_correction_topic, Header, self.gmapping_correction_header_callback, queue_size=10)
        self.cmd_vel_twist_subscriber = rospy.Subscriber(cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=1)
        self.map_subscriber = rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback, queue_size=1)
        self.scan_subscriber = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.on_shutdown(self.shutdown_callback)

    def loop(self):
        self.write_event(rospy.Time.now(), 'run_start')

        # check if it's time to finish
        r = rospy.Rate(20.0)
        while not rospy.is_shutdown() and not self.terminate:
            r.sleep()

        # check one last map snapshot
        if self.last_map_msg is not None:
            last_map_image_file_path = path.join(self.map_output_folder, "last_map.pgm")
            last_map_info_file_path = path.join(self.map_output_folder, "last_map_info.yaml")
            save_map_image(self.last_map_msg, last_map_image_file_path, last_map_info_file_path, self.map_free_threshold, self.map_occupied_threshold)

        self.write_event(rospy.Time.now(), 'supervisor_finished')

    def shutdown_callback(self):
        if not self.terminate:
            print_info("slam_benchmark_supervisor: asked to shutdown, terminating run")
            self.write_event(rospy.Time.now(), 'ros_shutdown')
            self.terminate = True

    def exploration_finished_callback(self, msg):
        if msg.data and not self.terminate:
            print_info("slam_benchmark_supervisor: explorer finished, terminating run")
            self.write_event(rospy.Time.now(), 'exploration_finished')
            self.terminate = True

    def run_timeout_callback(self, _):
        print_info("slam_benchmark_supervisor: terminating supervisor due to timeout, terminating run")
        self.write_event(rospy.Time.now(), 'run_timeout')
        self.terminate = True

    def ground_truth_pose_callback(self, odometry_msg):
        position = odometry_msg.pose.pose.position
        orientation = odometry_msg.pose.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, theta = tf.transformations.euler_from_quaternion(quaternion)

        if self.initial_ground_truth_pose is None:
            self.initial_ground_truth_pose = odometry_msg.pose.pose

        init_position = self.initial_ground_truth_pose.position
        with open(self.ground_truth_poses_file_path, 'a') as ground_truth_poses_file:
            ground_truth_poses_file.write("{t}, {x}, {y}, {theta}\n".format(t=odometry_msg.header.stamp.to_sec(),
                                                                            x=position.x - init_position.x,
                                                                            y=position.y - init_position.y,
                                                                            theta=theta))

    def cmd_vel_callback(self, twist_msg):
        with open(self.cmd_vel_twists_file_path, 'a') as cmd_vel_twists_file:
            cmd_vel_twists_file.write("{t}, {v_x}, {v_y}, {v_theta}\n".format(t=rospy.Time.now().to_sec(),
                                                                              v_x=twist_msg.linear.x,
                                                                              v_y=twist_msg.linear.y,
                                                                              v_theta=twist_msg.angular.z))

    def map_callback(self, occupancy_grid_msg):
        self.last_map_msg = occupancy_grid_msg

    def scan_callback(self, laser_scan_msg):
        with open(self.scans_file_path, 'a') as scans_file:
            scans_file.write("{t}, {angle_min}, {angle_max}, {angle_increment}, {range_min}, {range_max}, {ranges}\n".format(
                t=laser_scan_msg.header.stamp.to_sec(),
                angle_min=laser_scan_msg.angle_min,
                angle_max=laser_scan_msg.angle_max,
                angle_increment=laser_scan_msg.angle_increment,
                range_min=laser_scan_msg.range_min,
                range_max=laser_scan_msg.range_max,
                ranges=', '.join(map(str, laser_scan_msg.ranges))))

    def gmapping_correction_header_callback(self, header_msg):
        try:
            stamped_trans = self.tf_buffer.lookup_transform('map', 'base_link', header_msg.stamp, timeout=rospy.Duration.from_sec(0.1))
            q = stamped_trans.transform.rotation
            tf1_quaternion = [q.x, q.y, q.z, q.w]
            _, _, theta = tf.transformations.euler_from_quaternion(tf1_quaternion)

            with open(self.base_link_correction_poses_file_path, 'a') as base_link_correction_poses_file:
                base_link_correction_poses_file.write("FLASER 0 0.0 0.0 0.0 {x} {y} {theta} {t}\n".format(x=stamped_trans.transform.translation.x,
                                                                                                          y=stamped_trans.transform.translation.y,
                                                                                                          theta=theta,
                                                                                                          t=stamped_trans.header.stamp.to_sec()))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def write_base_link_poses_timer_callback(self, _):
        try:
            stamped_trans = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            q = stamped_trans.transform.rotation
            tf1_quaternion = [q.x, q.y, q.z, q.w]
            _, _, theta = tf.transformations.euler_from_quaternion(tf1_quaternion)

            with open(self.base_link_poses_file_path, 'a') as base_link_poses_file:
                base_link_poses_file.write("FLASER 0 0.0 0.0 0.0 {x} {y} {theta} {t}\n".format(x=stamped_trans.transform.translation.x,
                                                                                               y=stamped_trans.transform.translation.y,
                                                                                               theta=theta,
                                                                                               t=stamped_trans.header.stamp.to_sec()))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def ps_snapshot_timer_callback(self, _):
        ps_snapshot_file_path = path.join(self.ps_output_folder, "ps_{i:08d}.pkl".format(i=self.ps_snapshot_count))

        processes_dicts_list = list()
        for process in self.ps_processes:
            try:
                process_copy = copy.deepcopy(process.as_dict())  # get all information about the process
            except psutil.NoSuchProcess:  # processes may have died, causing this exception to be raised from psutil.Process.as_dict
                continue
            try:
                # delete uninteresting values
                del process_copy['connections']
                del process_copy['memory_maps']
                del process_copy['environ']

                processes_dicts_list.append(process_copy)
            except KeyError:
                pass

        with open(ps_snapshot_file_path, 'w') as ps_snapshot_file:
            pickle.dump(processes_dicts_list, ps_snapshot_file)

        self.ps_snapshot_count += 1

    def save_map_snapshot_timer_callback(self, _):

        if self.last_map_msg is None:
            return

        latest_map_image_file_path = path.join(self.map_output_folder, "map_{i:08d}.pgm".format(i=self.map_snapshot_count))
        latest_map_info_file_path = path.join(self.map_output_folder, "map_{i:08d}_info.yaml".format(i=self.map_snapshot_count))
        save_map_image(self.last_map_msg, latest_map_image_file_path, latest_map_info_file_path, self.map_free_threshold, self.map_occupied_threshold)

        steady_state_count = int(np.math.ceil(self.map_steady_state_period / self.map_snapshot_period))

        if self.map_snapshot_count > steady_state_count:
            previous_map_image_file_path = path.join(self.map_output_folder, "map_{i:08d}.pgm".format(i=self.map_snapshot_count - steady_state_count))

            if not map_changed(previous_map_image_file_path, latest_map_image_file_path, self.map_size_change_threshold, self.map_change_threshold):
                print_info("slam_benchmark_supervisor: map change lower than threshold, terminating run")
                self.write_event(rospy.Time.now(), 'no_map_change')
                self.terminate = True

        self.map_snapshot_count += 1

    def init_run_events_file(self):
        backup_file_if_exists(self.run_events_file_path)
        try:
            with open(self.run_events_file_path, 'w') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t='timestamp', event='event'))
        except IOError as e:
            rospy.logerr("slam_benchmark_supervisor.init_event_file: could not write header to run_events_file")
            rospy.logerr(e)

    def write_event(self, stamp, event):
        print_info("slam_benchmark_supervisor: t: {t}, event: {event}".format(t=stamp.to_sec(), event=str(event)))
        try:
            with open(self.run_events_file_path, 'a') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t=stamp.to_sec(), event=str(event)))
        except IOError as e:
            rospy.logerr("slam_benchmark_supervisor.write_event: could not write event to run_events_file: {t} {event}".format(t=stamp.to_sec(), event=str(event)))
            rospy.logerr(e)
