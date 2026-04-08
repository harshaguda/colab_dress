import cv2
import numpy as np
import rclpy
import time
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PoseArray, Pose


class DressNode(Node):
    def __init__(self):
        super().__init__('dress_node')

        # Subscription to pose estimates
        self.pose_subscription = self.create_subscription(
            PoseArray,
            '/pose_estimator/pose_3d',
            self.pose_callback,
            qos_profile_sensor_data)
        
        self.dmp_pub = self.create_publisher(PoseArray, 'dmp/arm_poses', 10)
        
        self.dressed = False
        self.index = 0
        self.poses_array = np.empty((20, 4, 3))

        #wait for 5 seconds to receive pose estimates before starting dressing
        self.get_logger().info("Waiting for pose estimates...")
        time.sleep(5)
        self.get_logger().info("Starting dressing process.")


    
    def pose_callback(self, msg: PoseArray):
        # Process the received PoseArray message
        # self.get_logger().info(f"Received {len(msg.poses)} poses.")
        curr_poses = []
        for i, pose in enumerate(msg.poses):
            curr_poses.append([pose.position.x, pose.position.y, pose.position.z])
            # self.get_logger().info(f"Pose {i}: Position - x: {pose.position.x}, y: {pose.position.y}, z: {pose.position.z}")  
            # self.get_logger().info(f"Pose array: {curr_poses}")     
        if not self.dressed:
            self.poses_array[self.index] = np.array(curr_poses)
            self.index += 1
            if self.index >= 19:
                self.index = 0
                pose_to_send = np.median(self.poses_array, axis=0)
                
                # Publish PoseArray for DMP
                out_msg = PoseArray()
                out_msg.header = Header()
                out_msg.header.frame_id = "base"
                out_msg.header.stamp = self.get_clock().now().to_msg()
                for pt in pose_to_send:
                    p = Pose()
                    p.position.x = float(pt[0])
                    p.position.y = float(pt[1])
                    p.position.z = float(pt[2])
                    out_msg.poses.append(p)
                self.dmp_pub.publish(out_msg)

                self.get_logger().info(f"Pose to send: {pose_to_send}")
                self.get_logger().info("Started Dressing. Stored initial poses.") 
                
                # Wait briefly to ensure message is published before shutdown
                time.sleep(0.5) 
                
                rclpy.shutdown()
                exit(0)
                


def main(args=None):
    rclpy.init(args=args)
    dress_node = DressNode()
    rclpy.spin(dress_node)
    dress_node.destroy_node()
    rclpy.shutdown()        

