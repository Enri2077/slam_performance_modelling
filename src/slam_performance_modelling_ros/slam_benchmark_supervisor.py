#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import time
import traceback
from collections import defaultdict
import copy
import pickle
import psutil
import os
from os import path
import numpy as np
import networkx as nx
import pandas as pd
import pyquaternion

import rospy
import tf2_ros
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Quaternion, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan

from performance_modelling_py.environment import ground_truth_map
from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error
from slam_toolbox_msgs.srv import SaveMap, SerializePoseGraph, SaveMapRequest, SerializePoseGraphRequest
from std_msgs.msg import String


class RunFailException(Exception):
    pass


def main():
    rospy.init_node('slam_benchmark_supervisor', anonymous=False)

    node = None

    # noinspection PyBroadException
    try:
        node = SlamBenchmarkSupervisor()
        node.start_run()
        rospy.spin()

    except KeyboardInterrupt:
        node.ros_shutdown_callback()
    except RunFailException as e:
        print_error(e)
    except Exception:
        print_error(traceback.format_exc())

    finally:
        if node is not None:
            node.end_run()
        if not rospy.is_shutdown():
            print_info("calling rospy signal_shutdown")
            rospy.signal_shutdown("run_terminated")


class SlamBenchmarkSupervisor:
    def __init__(self):

        # topics, services, actions, entities and frames names
        scan_topic = rospy.get_param('~scan_topic')
        scan_gt_topic = rospy.get_param('~scan_gt_topic')
        ground_truth_pose_topic = rospy.get_param('~ground_truth_pose_topic')
        save_map_service = rospy.get_param('~save_map_service')
        serialize_map_service = rospy.get_param('~serialize_map_service')
        estimated_pose_correction_topic = rospy.get_param('~estimated_pose_correction_topic')
        navigate_to_pose_action = rospy.get_param('~navigate_to_pose_action')
        self.fixed_frame = rospy.get_param('~fixed_frame')
        self.robot_base_frame = rospy.get_param('~robot_base_frame')
        self.robot_entity_name = rospy.get_param('~robot_entity_name')
        self.robot_radius = rospy.get_param('~robot_radius')
        self.fewer_nav_goals = rospy.get_param('~fewer_nav_goals')
        self.random_traversal_path = rospy.get_param('~random_traversal_path')
        self.environment_type = rospy.get_param('~environment_type')

        # file system paths
        self.run_output_folder = rospy.get_param('~run_output_folder')
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.ps_output_folder = path.join(self.benchmark_data_folder, "ps_snapshots")
        self.ground_truth_map_info_path = rospy.get_param('~ground_truth_map_info_path')
        self.output_map_file_path = path.join(self.benchmark_data_folder, "maps", "map")
        self.output_pose_graph_file_path = path.join(self.benchmark_data_folder, "maps", "posegraph")

        # run parameters
        run_timeout = rospy.get_param('~run_timeout')
        self.waypoint_timeout = rospy.get_param('~waypoint_timeout')
        self.estimated_pose_error_timeout = rospy.get_param('~estimated_pose_error_timeout')
        ps_snapshot_period = rospy.get_param('~ps_snapshot_period')
        write_estimated_poses_period = rospy.get_param('~write_estimated_poses_period')
        self.ps_pid_father = rospy.get_param('~pid_father')
        self.ps_processes = psutil.Process(self.ps_pid_father).children(recursive=True)  # list of processes children of the benchmark script, i.e., all ros nodes of the benchmark including this one
        self.ground_truth_map = ground_truth_map.GroundTruthMap(self.ground_truth_map_info_path)
        self.initial_pose_covariance_matrix = np.zeros((6, 6), dtype=float)
        self.initial_pose_covariance_matrix[0, 0] = rospy.get_param('~initial_pose_std_xy')**2
        self.initial_pose_covariance_matrix[1, 1] = rospy.get_param('~initial_pose_std_xy')**2
        self.initial_pose_covariance_matrix[5, 5] = rospy.get_param('~initial_pose_std_theta')**2
        self.goal_tolerance = rospy.get_param('~goal_tolerance')
        self.min_distance_traversal_path = rospy.get_param('~min_distance_traversal_path')

        # run variables
        self.run_started = False
        self.terminate = False
        self.ps_snapshot_count = 0
        self.received_first_scan = False
        self.latest_ground_truth_pose_msg = None
        self.initial_pose = None
        self.traversal_path_poses = None
        self.current_goal = None
        self.num_goals = None
        self.goal_sent_count = 0
        self.goal_succeeded_count = 0
        self.goal_failed_count = 0
        self.goal_rejected_count = 0
        self.estimated_pose_error = False
        self.first_estimated_pose_error_time = None

        # prepare folder structure
        if not path.exists(self.benchmark_data_folder):
            os.makedirs(self.benchmark_data_folder)

        if not path.exists(self.ps_output_folder):
            os.makedirs(self.ps_output_folder)

        # file paths for benchmark data
        self.estimated_poses_file_path = path.join(self.benchmark_data_folder, "estimated_poses.csv")
        self.estimated_correction_poses_file_path = path.join(self.benchmark_data_folder, "estimated_correction_poses.csv")
        self.ground_truth_poses_file_path = path.join(self.benchmark_data_folder, "ground_truth_poses.csv")
        self.scans_file_path = path.join(self.benchmark_data_folder, "scans.csv")
        self.scans_gt_file_path = path.join(self.benchmark_data_folder, "scans_gt.csv")
        self.run_events_file_path = path.join(self.benchmark_data_folder, "run_events.csv")
        self.init_run_events_file()

        # pandas dataframes for benchmark data
        self.estimated_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta'])
        self.estimated_correction_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'cov_x_x', 'cov_x_y', 'cov_y_y', 'cov_theta_theta'])
        if self.environment_type == 'simulation':
            self.ground_truth_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'v_x', 'v_y', 'v_theta'])
        else:
            self.ground_truth_poses_df = None

        # setup timers
        rospy.Timer(rospy.Duration.from_sec(run_timeout), self.run_timeout_callback)
        rospy.Timer(rospy.Duration.from_sec(ps_snapshot_period), self.ps_snapshot_timer_callback)
        rospy.Timer(rospy.Duration.from_sec(write_estimated_poses_period), self.write_estimated_pose_timer_callback)

        # setup service clients
        self.save_map_service_client = rospy.ServiceProxy(save_map_service, SaveMap)
        self.serialize_map_client = rospy.ServiceProxy(serialize_map_service, SerializePoseGraph)

        # setup buffers
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # setup publishers
        self.traversal_path_publisher = rospy.Publisher("~/traversal_path", Path, queue_size=1)

        # setup subscribers
        rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(scan_gt_topic, LaserScan, self.scan_gt_callback, queue_size=1)
        rospy.Subscriber(estimated_pose_correction_topic, PoseWithCovarianceStamped, self.estimated_pose_correction_callback, queue_size=1)
        if self.environment_type == 'simulation':
            rospy.Subscriber(ground_truth_pose_topic, Odometry, self.ground_truth_pose_callback, queue_size=1)

        # setup action clients
        self.navigate_to_pose_action_client = SimpleActionClient(navigate_to_pose_action, MoveBaseAction)

    def start_run(self):
        if self.environment_type == 'dataset':
            self.start_dataset_run()
        elif self.environment_type == 'simulation':
            self.start_simulation_run()
        else:
            raise RunFailException("environment_type [{}] is neither 'dataset' nor 'simulation'".format(self.environment_type))

    def start_dataset_run(self):
        print_info("preparing to start dataset run")
        self.write_event('run_start')
        self.run_started = True
        rospy.spin()
        self.write_event('run_completed')
        # self.save_map()  # TODO find way to save map and posegraph?

    def start_simulation_run(self):
        print_info("preparing to start simulation run")

        # wait to receive sensor data from the environment (e.g., a simulator may need time to startup)
        waiting_time = 0.0
        waiting_period = 0.5
        while not self.received_first_scan and not rospy.is_shutdown():
            time.sleep(waiting_period)
            waiting_time += waiting_period
            if waiting_time > 5.0:
                rospy.logwarn('still waiting to receive first sensor message from environment')
                waiting_time = 0.0

        # get deleaved reduced Voronoi graph from ground truth map
        voronoi_graph = self.ground_truth_map.deleaved_reduced_voronoi_graph(minimum_radius=2*self.robot_radius).copy()
        minimum_length_paths = nx.all_pairs_dijkstra_path(voronoi_graph, weight='voronoi_path_distance')
        minimum_length_costs = dict(nx.all_pairs_dijkstra_path_length(voronoi_graph, weight='voronoi_path_distance'))
        costs = defaultdict(dict)
        for i, paths_dict in minimum_length_paths:
            for j in paths_dict.keys():
                if i != j:
                    costs[i][j] = minimum_length_costs[i][j]

        # in case the graph has multiple unconnected components, remove the components smaller than 10% of the total graph
        too_small_voronoi_graph_components = list(filter(lambda component: len(component) < 0.1*len(voronoi_graph), nx.connected_components(voronoi_graph)))

        for graph_component in too_small_voronoi_graph_components:
            print_info("ignoring {} nodes from unconnected components in the Voronoi graph".format(len(graph_component)))
            voronoi_graph.remove_nodes_from(graph_component)

        if len(voronoi_graph.nodes) < 2:
            self.write_event('insufficient_number_of_nodes_in_deleaved_reduced_voronoi_graph')
            raise RunFailException("insufficient number of nodes in deleaved_reduced_voronoi_graph, can not generate traversal path")

        # get greedy path traversing the whole graph starting from a random node
        traversal_path_indices = list()
        if self.random_traversal_path:
            current_node = random.choice(list(voronoi_graph.nodes))
        else:
            current_node = list(voronoi_graph.nodes)[0]
        nodes_queue = set(nx.node_connected_component(voronoi_graph, current_node))
        while len(nodes_queue):
            candidates = list(filter(lambda node_cost: node_cost[0] in nodes_queue, costs[current_node].items()))
            candidate_nodes, candidate_costs = zip(*candidates)
            next_node = candidate_nodes[int(np.argmin(candidate_costs))]
            traversal_path_indices.append(next_node)
            current_node = next_node
            nodes_queue.remove(next_node)

        # convert path of nodes to list of poses
        self.traversal_path_poses = list()
        for node_index in traversal_path_indices:
            pose = Pose()
            pose.position.x, pose.position.y = voronoi_graph.nodes[node_index]['vertex']
            q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
            pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)

            if self.fewer_nav_goals and len(self.traversal_path_poses) > 0:
                prev_position = self.traversal_path_poses[-1].position
                distance_from_prev_node = np.sqrt((pose.position.x - prev_position.x) ** 2 + (pose.position.y - prev_position.y) ** 2)
                if distance_from_prev_node < self.min_distance_traversal_path:
                    continue

            self.traversal_path_poses.append(pose)

        # add the reversed path to make sure all parts of the map are visited in both directions
        reversed_traversal_path_poses = copy.deepcopy(self.traversal_path_poses[::-1])
        for pose in reversed_traversal_path_poses:
            q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
            pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
        self.traversal_path_poses += reversed_traversal_path_poses

        if len(self.traversal_path_poses) < 2:
            self.write_event('insufficient_number_of_poses_in_traversal_path')
            raise RunFailException("insufficient number of poses in traversal path, can not send goal")

        self.publish_traversal_path()

        self.num_goals = len(self.traversal_path_poses)

        time.sleep(5.0)
        self.write_event('run_start')
        self.run_started = True

        # send goals
        for traversal_path_pose in self.traversal_path_poses:
            print_info("goal {} / {}".format(self.goal_sent_count + 1, self.num_goals))

            if rospy.is_shutdown():
                break

            if not self.navigate_to_pose_action_client.wait_for_server(timeout=rospy.Duration.from_sec(5.0)):
                self.write_event('failed_to_communicate_with_navigation_node')
                raise RunFailException("navigate_to_pose action server not available")

            goal_msg = MoveBaseGoal()
            goal_msg.target_pose.header.stamp = rospy.Time.now()
            goal_msg.target_pose.header.frame_id = self.fixed_frame
            goal_msg.target_pose.pose = traversal_path_pose

            self.navigate_to_pose_action_client.send_goal(goal_msg)
            self.write_event('target_pose_set')
            self.goal_sent_count += 1

            if not self.navigate_to_pose_action_client.wait_for_result(timeout=rospy.Duration.from_sec(self.waypoint_timeout)):
                self.write_event('waypoint_timeout')
                self.write_event('supervisor_finished')
                raise RunFailException("waypoint_timeout")

            if self.navigate_to_pose_action_client.get_state() == GoalStatus.SUCCEEDED:
                goal_position = goal_msg.target_pose.pose.position
                current_position = self.latest_ground_truth_pose_msg.pose.pose.position
                distance_from_goal = np.sqrt((goal_position.x - current_position.x) ** 2 + (goal_position.y - current_position.y) ** 2)
                if distance_from_goal < self.goal_tolerance:
                    self.write_event('target_pose_reached')
                    self.goal_succeeded_count += 1
                else:
                    print_error("goal status succeeded but current position farther from goal position than tolerance")
                    self.write_event('target_pose_not_reached')
                    self.goal_failed_count += 1
            else:
                print_info('navigation action failed with status {}, {}'.format(self.navigate_to_pose_action_client.get_state(), self.navigate_to_pose_action_client.get_goal_status_text()))
                self.write_event('target_pose_not_reached')
                self.goal_failed_count += 1
                if self.goal_failed_count < 10:
                    self.traversal_path_poses.append(traversal_path_pose)
                    self.publish_traversal_path()

            rospy.sleep(1.0)

        # if all goals have been sent end the run
        self.write_event('run_completed')
        self.save_map()
        rospy.signal_shutdown("run_completed")

    def publish_traversal_path(self):
        traversal_path_msg = Path()
        traversal_path_msg.header.frame_id = self.fixed_frame
        traversal_path_msg.header.stamp = rospy.Time.now()
        for traversal_pose in self.traversal_path_poses:
            traversal_pose_stamped = PoseStamped()
            traversal_pose_stamped.header = traversal_path_msg.header
            traversal_pose_stamped.pose = traversal_pose
            traversal_path_msg.poses.append(traversal_pose_stamped)
        self.traversal_path_publisher.publish(traversal_path_msg)

    def save_map(self):
        print_info("sending save map request")
        if not path.exists(path.dirname(self.output_map_file_path)):
            os.makedirs(path.dirname(self.output_map_file_path))
        if not path.exists(path.dirname(self.output_pose_graph_file_path)):
            os.makedirs(path.dirname(self.output_pose_graph_file_path))

        try:
            self.save_map_service_client.wait_for_service(timeout=1.0)
            self.save_map_service_client.call(SaveMapRequest(name=String(data=self.output_map_file_path)))
        except rospy.ROSException, rospy.ServiceException:
            print_info("Service not available: {}".format(self.save_map_service_client.resolved_name))
            self.write_event('failed_to_save_map')

        try:
            self.serialize_map_client.wait_for_service(timeout=1.0)
            self.serialize_map_client.call(SerializePoseGraphRequest(filename=self.output_pose_graph_file_path))
        except rospy.ROSException, rospy.ServiceException:
            print_info("Service not available: {}".format(self.serialize_map_client.resolved_name))
            self.write_event('failed_to_serialize_pose_graph')

        self.write_event('save_map_request_sent')

    def ros_shutdown_callback(self):
        """
        This function is called when the node receives an interrupt signal (KeyboardInterrupt).
        """
        print_info("asked to shutdown, terminating run")
        self.write_event('ros_shutdown')
        self.write_event('supervisor_finished')

    def end_run(self):
        """
        This function is called after the run has completed, whether the run finished correctly, or there was an exception.
        The only case in which this function is not called is if an exception was raised from self.__init__
        """
        self.estimated_poses_df.to_csv(self.estimated_poses_file_path, index=False)
        self.estimated_correction_poses_df.to_csv(self.estimated_correction_poses_file_path, index=False)
        if self.environment_type == 'simulation':
            self.ground_truth_poses_df.to_csv(self.ground_truth_poses_file_path, index=False)

    def run_timeout_callback(self, _):
        print_error("terminating supervisor due to timeout, terminating run")
        self.write_event('run_timeout')
        self.write_event('supervisor_finished')
        rospy.signal_shutdown("run_timeout")

    def scan_callback(self, laser_scan_msg):
        self.received_first_scan = True
        if not self.run_started:
            return

        msg_time = laser_scan_msg.header.stamp.to_sec()
        with open(self.scans_file_path, 'a') as scans_file:
            scans_file.write("{t}, {angle_min}, {angle_max}, {angle_increment}, {range_min}, {range_max}, {ranges}\n".format(
                t=msg_time,
                angle_min=laser_scan_msg.angle_min,
                angle_max=laser_scan_msg.angle_max,
                angle_increment=laser_scan_msg.angle_increment,
                range_min=laser_scan_msg.range_min,
                range_max=laser_scan_msg.range_max,
                ranges=', '.join(map(str, laser_scan_msg.ranges))))

    def scan_gt_callback(self, laser_scan_msg):
        if not self.run_started:
            return

        msg_time = laser_scan_msg.header.stamp.to_sec()
        with open(self.scans_gt_file_path, 'a') as scans_file:
            scans_file.write("{t}, {angle_min}, {angle_max}, {angle_increment}, {range_min}, {range_max}, {ranges}\n".format(
                t=msg_time,
                angle_min=laser_scan_msg.angle_min,
                angle_max=laser_scan_msg.angle_max,
                angle_increment=laser_scan_msg.angle_increment,
                range_min=laser_scan_msg.range_min,
                range_max=laser_scan_msg.range_max,
                ranges=', '.join(map(str, laser_scan_msg.ranges))))

    def write_estimated_pose_timer_callback(self, _):
        if not self.run_started:
            return

        # noinspection PyBroadException
        try:
            transform_msg = self.tf_buffer.lookup_transform(self.fixed_frame, self.robot_base_frame, rospy.Time())
            orientation = transform_msg.transform.rotation
            theta, _, _ = pyquaternion.Quaternion(x=orientation.x, y=orientation.y, z=orientation.z, w=orientation.w).yaw_pitch_roll

            self.estimated_poses_df = self.estimated_poses_df.append({
                't': transform_msg.header.stamp.to_sec(),
                'x': transform_msg.transform.translation.x,
                'y': transform_msg.transform.translation.y,
                'theta': theta
            }, ignore_index=True)
            self.estimated_pose_error = False
            self.first_estimated_pose_error_time = None

        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        except:
            print_error(traceback.format_exc())
            if not self.estimated_pose_error:
                self.estimated_pose_error = True
                self.first_estimated_pose_error_time = rospy.Time.now()

            if rospy.Time.now() - self.first_estimated_pose_error_time > rospy.Duration.from_sec(self.estimated_pose_error_timeout):
                print_error("terminating supervisor due to estimated pose error, terminating run")
                self.write_event('estimated_pose_error')
                self.write_event('supervisor_finished')
                rospy.signal_shutdown("estimated_pose_error")

    def estimated_pose_correction_callback(self, pose_with_covariance_msg):
        if not self.run_started:
            return

        orientation = pose_with_covariance_msg.pose.pose.orientation
        theta, _, _ = pyquaternion.Quaternion(x=orientation.x, y=orientation.y, z=orientation.z, w=orientation.w).yaw_pitch_roll
        covariance_mat = np.array(pose_with_covariance_msg.pose.covariance).reshape(6, 6)

        self.estimated_correction_poses_df = self.estimated_correction_poses_df.append({
            't': pose_with_covariance_msg.header.stamp.to_sec(),
            'x': pose_with_covariance_msg.pose.pose.position.x,
            'y': pose_with_covariance_msg.pose.pose.position.y,
            'theta': theta,
            'cov_x_x': covariance_mat[0, 0],
            'cov_x_y': covariance_mat[0, 1],
            'cov_y_y': covariance_mat[1, 1],
            'cov_theta_theta': covariance_mat[5, 5]
        }, ignore_index=True)

    def ground_truth_pose_callback(self, odometry_msg):
        self.latest_ground_truth_pose_msg = odometry_msg
        if not self.run_started:
            return

        orientation = odometry_msg.pose.pose.orientation
        theta, _, _ = pyquaternion.Quaternion(x=orientation.x, y=orientation.y, z=orientation.z, w=orientation.w).yaw_pitch_roll

        self.ground_truth_poses_df = self.ground_truth_poses_df.append({
            't': odometry_msg.header.stamp.to_sec(),
            'x': odometry_msg.pose.pose.position.x,
            'y': odometry_msg.pose.pose.position.y,
            'theta': theta,
            'v_x': odometry_msg.twist.twist.linear.x,
            'v_y': odometry_msg.twist.twist.linear.y,
            'v_theta': odometry_msg.twist.twist.angular.z,
        }, ignore_index=True)

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
        try:
            with open(ps_snapshot_file_path, 'wb') as ps_snapshot_file:
                pickle.dump(processes_dicts_list, ps_snapshot_file)
        except TypeError:
            print_error(traceback.format_exc())

        self.ps_snapshot_count += 1

    def init_run_events_file(self):
        backup_file_if_exists(self.run_events_file_path)
        try:
            with open(self.run_events_file_path, 'w') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t='timestamp', event='event'))
        except IOError:
            rospy.logerr("slam_benchmark_supervisor.init_event_file: could not write header to run_events_file")
            rospy.logerr(traceback.format_exc())

    def write_event(self, event):
        t = rospy.Time.now().to_sec()
        print_info("t: {t}, event: {event}".format(t=t, event=str(event)))
        try:
            with open(self.run_events_file_path, 'a') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t=t, event=str(event)))
        except IOError:
            rospy.logerr("slam_benchmark_supervisor.write_event: could not write event to run_events_file: {t} {event}".format(t=t, event=str(event)))
            rospy.logerr(traceback.format_exc())
