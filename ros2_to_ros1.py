import os 

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

from rosbags.rosbag1 import Writer
from rosbags.serde import cdr_to_ros1, serialize_cdr
from rosbags.typesys.types import std_msgs__msg__String as String

from rosbags.interfaces import Connection, ConnectionExtRosbag1, ConnectionExtRosbag2
from rosbags.rosbag1 import Reader as Reader1
from rosbags.rosbag1 import ReaderError as ReaderError1
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag1 import WriterError as WriterError1
from rosbags.rosbag2 import Reader as Reader2
from rosbags.rosbag2 import ReaderError as ReaderError2
from rosbags.rosbag2 import Writer as Writer2
from rosbags.rosbag2 import WriterError as WriterError2
from rosbags.serde import cdr_to_ros1, ros1_to_cdr
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.msg import generate_msgdef


ROS2_PATH = "resources/data/rosbag2_2023_11_21-15_04_43"
ROS1_PATH = "resources/data/rosbag2_2023_11_21-15_04_43.bag"


if os.path.exists(ROS1_PATH):
    os.remove(ROS1_PATH)

target_topics = {
    "/zed2/zed_node/left/image_rect_color/compressed": 
        "/realsense_gripper/color/image_raw/compressed",

    "/zed2i_front/zed_node/rgb/image_rect_color/compressedDepth":
        "/realsense_gripper/aligned_depth_to_color/image_raw",
    
    "/zed2i_front/zed_node/depth/camera_info":
        "/realsense_gripper/aligned_depth_to_color/camera_info",
}


def downgrade_connection(rconn: Connection, new_topic) -> Connection:
    """Convert rosbag2 connection to rosbag1 connection.

    Args:
        rconn: Rosbag2 connection.

    Returns:
        Rosbag1 connection.

    """
    assert isinstance(rconn.ext, ConnectionExtRosbag2)
    msgdef, md5sum = generate_msgdef(rconn.msgtype)
    return Connection(
        rconn.id,
        new_topic,
        rconn.msgtype,
        msgdef,
        md5sum,
        -1,
        ConnectionExtRosbag1(
            None,
            int('durability: 1' in rconn.ext.offered_qos_profiles),
        ),
        None,
    )


with Reader2(ROS2_PATH) as reader, Writer1(ROS1_PATH) as writer:
    connmap: dict[int, Connection] = {}
    connections = [
        x for x in reader.connections
        if x.topic in target_topics
    ]
    if not connections:
        raise ValueError('No connections left for conversion.')
    for rconn in connections:
        print(f'Ros2: "{rconn.topic}" -> Ros1 "{target_topics[rconn.topic]}"')
        candidate = downgrade_connection(rconn, target_topics[rconn.topic])
        assert isinstance(candidate.ext, ConnectionExtRosbag1)
        for conn in writer.connections:
            assert isinstance(conn.ext, ConnectionExtRosbag1)
            if (
                conn.topic == candidate.topic and conn.digest == candidate.digest and
                conn.ext.latching == candidate.ext.latching
            ):
                break
        else:
            conn = writer.add_connection(
                candidate.topic,
                candidate.msgtype,
                candidate.msgdef,
                candidate.digest,
                candidate.ext.callerid,
                candidate.ext.latching,
            )
        connmap[rconn.id] = conn

    for rconn, timestamp, data in reader.messages(connections=connections):
        data = cdr_to_ros1(data, rconn.msgtype)
        writer.write(connmap[rconn.id], timestamp, data)
