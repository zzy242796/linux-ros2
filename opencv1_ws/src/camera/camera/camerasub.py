#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('camerasub')
        self.bridge = CvBridge()
        qos = QoSProfile(depth=10)
        self.mat = None
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos)
        

    def image_callback(self, msg):
        # 将ROS消息转换为OpenCV格式
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # # 显示图像
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)  # 等待1毫秒，以便显示窗口能够更新

def main(args=None):
    rclpy.init(args=args)
    camerasub_node = ImageSubscriber()
    rclpy.spin(camerasub_node)
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
