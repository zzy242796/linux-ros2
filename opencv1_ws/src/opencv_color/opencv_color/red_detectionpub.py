#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from rclpy.qos import QoSProfile

global center_x, center_y, delta_x, text
center_x = 0.0
center_y = 0.0
delta_x = 0.0
text = 'no'

class RedDetectionNode(Node):
    def __init__(self):
        super().__init__('red_detectionpub')
        self.bridge = CvBridge()
        qos = QoSProfile(depth=10)
        
        # 创建发布器
        self.image_pub = self.create_publisher(Image, 'red_detection_image', 10)
        self.center_pub = self.create_publisher(Point, 'red_center_point', 10)

        # 创建订阅器
        self.subscription = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            qos
        )

        # 设置HSV颜色空间的红色阈值（提高阈值）
        self.lower_red = np.array([0, 150, 100])  # H=0, S=150, V=100
        self.upper_red = np.array([10, 255, 255])  # H=10, S=255, V=255
        self.lower_red2 = np.array([160, 150, 100])  # H=160, S=150, V=100
        self.upper_red2 = np.array([180, 255, 255])  # H=180, S=255, V=255

        # 设置HSV颜色空间的蓝色阈值
        self.lower_blue = np.array([100, 150, 100])  # H=100, S=150, V=100
        self.upper_blue = np.array([140, 255, 255])  # H=140, S=255, V=255
        self.lower_blue2 = np.array([90, 150, 100])   # H=90, S=150, V=100
        self.upper_blue2 = np.array([100, 255, 255])  # H=100, S=255, V=255

    def image_callback(self, msg):
        global center_x, center_y, delta_x, text
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        decoded_objects = decode(frame)

        for obj in decoded_objects:
            # 绘制矩形框
            cv2.rectangle(frame, (obj.rect.left, obj.rect.top), (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height), (0, 255, 0), 2)
        
            # 显示二维码数据
            text = obj.data.decode("utf-8")
            cv2.putText(frame, text, (obj.rect.left, obj.rect.top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if text == 'blue':
            # 创建蓝色掩膜
            mask1 = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            mask2 = cv2.inRange(hsv, self.lower_blue2, self.upper_blue2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # 创建红色掩膜
            mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作以去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果找到轮廓，处理最大轮廓
        if contours:
            self.process_contours(frame, contours)
        else:
            center_x = 0.0
            center_y = 0.0
            delta_x = 0.0
            self.publish_results(frame, center_x, center_y,delta_x)

        cv2.imshow('Red Detection', frame)
        if self.check_exit_key():
            cv2.destroyAllWindows()

    def process_contours(self, frame, contours):
        global center_x, center_y, delta_x
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        center_x = float(x + w // 2)
        center_y = float(y + h // 2)

         # 获取图像的宽高
        height, width, _ = frame.shape
        image_center_x = float(width // 2)  # 图像中心横坐标

        # 计算中心点差值
        delta_x = center_x - image_center_x
        # self.get_logger().info(f"Delta X: {delta_x}")

        # 在框和中心添加绘图
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        cv2.line(frame, (int(center_x) - 5, int(center_y)), (int(center_x) + 5, int(center_y)), (0, 255, 0), 1)
        cv2.line(frame, (int(center_x), int(center_y) - 5), (int(center_x), int(center_y) + 5), (0, 255, 0), 1)

        # 发布图像和中心点坐标
        self.publish_results(frame, center_x, center_y,delta_x)

    def publish_results(self, frame, center_x, center_y,delta_x):
        msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_pub.publish(msg)

        center_msg = Point()
        center_msg.x = center_x
        center_msg.y = center_y
        center_msg.z = delta_x # 假设为2D坐标
        self.center_pub.publish(center_msg)
        self.get_logger().info(f"Published center point: ({int(center_x)}, {int(center_y)},{delta_x})")

    def check_exit_key(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False

def main(args=None):
    rclpy.init(args=args)
    red_detectionpub_node = RedDetectionNode()
    try:
        rclpy.spin(red_detectionpub_node)
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
