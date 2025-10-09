import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import cv2

class PublishImage(Node):
    
    def __init__(self):
        super().__init__('publish_image')
        self.publisher_ = self.create_publisher(Image, 'camera/image', qos_profile_sensor_data)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, 'camera/camera_info', qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.cap = cv2.VideoCapture(5)  # Open the default camera
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the image to ROS Image message
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera_frame'
            self.publisher_.publish(img_msg)

            # Publish dummy CameraInfo
            camera_info_msg = CameraInfo()
            camera_info_msg.header = img_msg.header
            camera_info_msg.width = frame.shape[1]
            camera_info_msg.height = frame.shape[0]
            # Fill in other CameraInfo fields as needed
            self.camera_info_publisher_.publish(camera_info_msg)

def main(args=None):
    rclpy.init(args=args)
    publish_image = PublishImage()
    rclpy.spin(publish_image)
    publish_image.cap.release()
    publish_image.destroy_node()
    rclpy.shutdown()