#!/bin/bash

docker build \
  --build-arg user_uid=`id -u` \
  --build-arg user_name=$USER \
  --build-arg user_gid=`id -g` \
  --build-arg user_group=`id -g -n $USER` \
  --no-cache \
  -t ${USER}/slam_benchmark:v15 .
