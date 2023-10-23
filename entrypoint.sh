cd /sources/catkin_ws/
/opt/ros/noetic/bin/catkin_make
source /sources/catkin_ws/devel/setup.bash

echo "Run roscore"
roscore &
sleep 2
echo "Run bag play"
rosbag play /sources/test_data.bag -i
sleep 2

echo "Run bot_sort_node.py"
python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/bot_sort_node.py -vis &
sleep 2

echo "Run rviz"
rosrun rviz rviz -d /sources/rviz_conf.rviz &
sleep 2

rosbag play /sources/test_data.bag 
exec /bin/bash 