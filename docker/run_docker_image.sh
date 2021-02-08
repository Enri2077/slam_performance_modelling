#!/bin/bash

# change USER in /home/${USER} if the host user name is not the same as the container user name

if [ $# -eq 2 ]
  then
    echo "container id: $1"
    echo "cpu set: $2"
    docker run -ti --rm \
        --name ${USER}_slam_benchmark_v15_$1 \
        --user $(id -u):$(id -g) \
        --cpuset-cpus "$2" \
        -v ~/datasets/private/performance_modelling:/home/${USER}/ds/performance_modelling \
        -v ~/datasets/private/performance_modelling/ros_logs:/home/${USER}/.ros/log \
        ${USER}/slam_benchmark:v15
  else
    echo "usage: $0 container_id cpu_set"
fi
