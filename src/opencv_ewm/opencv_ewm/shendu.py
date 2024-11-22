import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class DepthReceiver(Node):
    def __init__(self):
        super().__init__('depth_receiver')
        # 创建订阅者，订阅深度值
        self.subscription = self.create_subscription(
            Float32,
            'depth_value',
            self.depth_callback,
            10)  # QoS参数设置

    def depth_callback(self, msg):
        depth_value = int(msg.data)  # 将消息数据转换为整型
        # 打印接收到的深度值
        self.get_logger().info(f'Received depth value: {msg.data:.2f} mm')

def main(args=None):
    rclpy.init(args=args)
    depth_receiver = DepthReceiver()
    rclpy.spin(depth_receiver)
    depth_receiver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
