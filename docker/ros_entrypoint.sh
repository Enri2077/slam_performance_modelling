#!/bin/bash
set -e

# setup ros environment and workspace
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/w/catkin_ws/devel/setup.bash

exec "$@"

