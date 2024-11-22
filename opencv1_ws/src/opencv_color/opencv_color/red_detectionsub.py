#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class RedDetectionListenerNode(Node):
    def __init__(self):
        super().__init__('red_detectionsub')
        
        # 创建一个订阅者，订阅geometry_msgs/Point类型的消息
        self.subscription = self.create_subscription(Point, 'red_center_point', self.listener_callback, 10)
        self.subscription  # 防止未命名变量的警告
        
        # 打印日志
        self.get_logger().info('Red detection listener node is ready.')

    def listener_callback(self, msg):
        # 定义回调函数，当接收到消息时执行
        self.get_logger().info('Received center point: x = %f, y = %f, z = %f' % (msg.x, msg.y, msg.z))

def main(args=None):
    rclpy.init(args=args)
    red_detectionsub_node = RedDetectionListenerNode()
    rclpy.spin(red_detectionsub_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()