#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camerapub')
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, 'cameraimg', 10)
        self.timer = self.create_timer(0.5, self.time_callback)

    def time_callback(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        self.get_logger().info('camera ok!')

        while True:
            # 读取摄像头的每一帧
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Failed to capture image')
                return

            msg =self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp =self.get_clock().now().to_msg()
            self.image_publisher.publish(msg)
            
def main(args=None):
    rclpy.init(args=args)
    camerapub_node = CameraNode()
    rclpy.spin(camerapub_node)
    camerapub_node.cap.release()
    rclpy.shutdown()

if __name__ == '__main__':
    main()