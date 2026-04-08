import yaml, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose

INPUT_FILE = '/home/hguda/colab_dress_ws/traj.txt'
TOPIC = '/cartesian_trajectory'

with open(INPUT_FILE, 'r') as f:
    docs = [doc for doc in yaml.safe_load_all(f) if doc]

if not docs:
    raise RuntimeError(f'No YAML document found in {INPUT_FILE}')

data = docs[0]

rclpy.init()
node = Node('posearray_once_pub')
pub = node.create_publisher(PoseArray, TOPIC, 10)

msg = PoseArray()
msg.header.frame_id = data['header']['frame_id']
msg.header.stamp.sec = int(data['header']['stamp']['sec'])
msg.header.stamp.nanosec = int(data['header']['stamp']['nanosec'])

for p in data['poses']:
    q = Pose()
    q.position.x = float(p['position']['x'])
    q.position.y = float(p['position']['y'])
    q.position.z = float(p['position']['z'])
    q.orientation.x = float(p['orientation']['x'])
    q.orientation.y = float(p['orientation']['y'])
    q.orientation.z = float(p['orientation']['z'])
    q.orientation.w = float(p['orientation']['w'])
    msg.poses.append(q)

pub.publish(msg)
rclpy.spin_once(node, timeout_sec=0.2)
node.destroy_node()
rclpy.shutdown()
print(f'Published {TOPIC} once')