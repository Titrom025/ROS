source /opt/ros/noetic/setup.bash

cd /sources/catkin_ws/
/opt/ros/noetic/bin/catkin_make
source /sources/catkin_ws/devel/setup.bash

rosparam set use_sim_time true

echo "Run roscore"
roscore &
sleep 1
echo "Run bag play"
rosbag play /sources/test_data.bag -i
sleep 1

echo "Run bot_sort_node.py"
python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/bot_sort_node.py -vis &
sleep 5

echo "Run tracker_3d_node.py"
python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/tracker_3d_node.py -vis &
sleep 2

echo "Run rviz"
rosrun rviz rviz -d /sources/rviz_conf.rviz &
sleep 5

# rosbag play /sources/test_data.bag
# rosbag play /sources/test_data.bag -r 0.5 -s 7
rosbag play /sources/test_data.bag -r 0.5
exec /bin/bash 