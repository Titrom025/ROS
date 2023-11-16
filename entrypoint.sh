source /opt/ros/noetic/setup.bash
cd /sources/catkin_ws/src/openseed_src/openseed/body/encoder/ops
sh make.sh

cd /sources/catkin_ws/
/opt/ros/noetic/bin/catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3.8 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.8m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8m.so

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

python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/text_query_generation_server.py &
while :
do
    if rostopic list -v | grep -q "/segmentation_labels \[husky_tidy_bot_cv/Categories\] 1 publisher"; then
        echo "Found /segmentation_labels with 1 publisher"
        break
    else
        echo "Waiting for /segmentation_labels..."
        sleep 5
    fi
done

python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/openseed_node.py -vis &
while :
do
    if rostopic list -v | grep -q "/segmentation_vis \[sensor_msgs/Image\] 1 publisher"; then
        echo "Found /segmentation_vis with 1 publisher"
        break
    else
        echo "Waiting for /segmentation_vis..."
        sleep 5
    fi
done


# rosbag play /sources/test_data.bag
# rosbag play /sources/test_data.bag -r 0.5 -s 7
rosbag play /sources/test_data.bag -r 0.5
exec /bin/bash 