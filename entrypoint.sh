source /opt/ros/noetic/setup.bash
source /sources/catkin_ws/devel/setup.bash

# Bag 1
export FIXED_FRAME=local_map_lidar
export TARGET_BAG=/resources/data/test_data.bag
export RVIZ_CONF=/resources/data/testbag_full.rviz
export DEPTH_TOPIC=/realsense_gripper/aligned_depth_to_color/image_raw
export IMAGE_TOPIC=/realsense_gripper/color/image_raw/compressed

# Bag 2
# export FIXED_FRAME=front_bumper_link
# export TARGET_BAG=/resources/data/2023-08-29-11-02-40_0.bag
# export RVIZ_CONF=/resources/data/rviz_ros_data290823.rviz
# export DEPTH_TOPIC=/realsense_gripper/aligned_depth_to_color/image_raw
# export IMAGE_TOPIC=/realsense_gripper/color/image_raw/compressed

# Bag 3
# export FIXED_FRAME=local_map_lidar
# export TARGET_BAG=/resources/data/rosbag2_2023_11_21-15_04_43.bag
# export RVIZ_CONF=/resources/data/testbag_full.rviz
# export DEPTH_TOPIC=/realsense_gripper/aligned_depth_to_color/image_raw
# export IMAGE_TOPIC=/realsense_gripper/color/image_raw/compressed

# Bag 4
# export IMAGE_TOPIC=/zedm_front/zed_node/left/image_rect_color/compressed
# export FIXED_FRAME=local_map_lidar
# export TARGET_BAG=/resources/data/2023-12-22-16-42-22.bag
# export RVIZ_CONF=/resources/data/2023-12-22-16-42-22.rviz
# export DEPTH_TOPIC=/zedm_front/zed_node/depth/depth_registered



# export RVIZ_CONF=/resources/data/2023-12-22-16-42-22_v2.rviz
# export RVIZ_CONF=/resources/data/2023-12-22-16-42-22_test.rviz
# export RVIZ_CONF=/resources/data/2023-12-22-16-42-22_test_v2.rviz

# export RVIZ_CONF=/resources/data/testbag_full.rviz
# export DEPTH_TOPIC=/zedm_front/zed_node/depth/depth_registered


echo "Run roscore"
roscore &
sleep 1

echo "Run bag play"
rosbag play $TARGET_BAG -i
sleep 1

echo "Run bot_sort_node.py"
python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/bot_sort_node.py -vis &
sleep 1

echo "Run tracker_3d_node.py"
python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/tracker_3d_node.py -vis &
sleep 1

echo "Run rviz"
rosrun rviz rviz -d $RVIZ_CONF &
sleep 1

python /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/text_query_generation_server.py &
while :
do
    if rostopic list -v | grep -q "/segmentation_labels \[husky_tidy_bot_cv/Categories\] 1 publisher"; then
        echo "Found /segmentation_labels with 1 publisher"
        break
    else
        echo "Waiting for /segmentation_labels..."
        sleep 1
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
        sleep 1
    fi
done


rosbag play $TARGET_BAG -r 0.5 -s 5
# rosbag play $TARGET_BAG -r 0.5 -s 5 -u 5
exec /bin/bash 