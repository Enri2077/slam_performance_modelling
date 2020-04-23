mkdir -p ~/w/catkin_ws/src
cd ~/w/catkin_ws/src

git clone https://github.com/Enri2077/performance_modelling_ros.git performance_modelling
git clone https://github.com/Enri2077/stage_ros.git
git clone https://github.com/Enri2077/slam_gmapping.git
git clone https://github.com/Enri2077/openslam_gmapping.git
git clone https://github.com/Enri2077/m-explore.git

cd ~/w/catkin_ws
catkin build

