import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # 替换your_package为你的包名

class TextSubscriber(Node):
    def __init__(self):
        super().__init__('erweimasub')
        self.subscription = self.create_subscription(
            String,
            'qr_code_text',
            self.listener_callback,
            10)
        self.subscription  # 防止未定义变量警告
        self.get_logger().info('wait qr......')


    def listener_callback(self, msg):
        if msg.data == 'successful!!!':
            self.get_logger().info('Received: "%s"' % msg.data)
        else:
            self.get_logger().info('Received: no' )

def main(args=None):
    rclpy.init(args=args)
    erweimasub_node = TextSubscriber()
    rclpy.spin(erweimasub_node)
    # text_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()