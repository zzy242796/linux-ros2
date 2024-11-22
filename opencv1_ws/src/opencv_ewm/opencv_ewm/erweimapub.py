#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from pyzbar.pyzbar import decode
from std_msgs.msg import String
import numpy as np

class QRCodeScanner(Node):
    def __init__(self):
        super().__init__('erweimapub')
        self.bridge = CvBridge()
        self.image_publisher_ = self.create_publisher(Image, 'qr_code_detection', 10)
        self.text_publisher_ = self.create_publisher(String, 'qr_code_text', 10)
        self.timer = self.create_timer(0.5, self.scan_qr_code)

    def scan_qr_code(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        self.get_logger().info('camera ok!')
        self.get_logger().info('wait qr.......')


        while True:
            text = 'no'
            # 读取摄像头的每一帧
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Failed to capture image')
                return

            # 解码二维码
            decoded_objects = decode(frame)

            # 遍历解码对象
            for obj in decoded_objects:
                # 绘制矩形框
                cv2.rectangle(frame, (obj.rect.left, obj.rect.top), (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height), (0, 255, 0), 2)
            
                # 显示二维码数据
                text = obj.data.decode("utf-8")
                cv2.putText(frame, text, (obj.rect.left, obj.rect.top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 发布解码的文本信息
            text_msg = String()
            text_msg.data = text
            self.text_publisher_.publish(text_msg)
            self.get_logger().info('Publishing QR code text: "%s"' % text)

            # 显示结果
            cv2.imshow("QR Code Scanner", frame)

            # 转换图像格式
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.image_publisher_.publish(msg)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放摄像头
        # cap.release()
        # 关闭所有OpenCV窗口
        # cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    erweimapub_node = QRCodeScanner()
    rclpy.spin(erweimapub_node)
    erweimapub_node.cap.release()
    rclpy.shutdown()

if __name__ == '__main__':
    main()