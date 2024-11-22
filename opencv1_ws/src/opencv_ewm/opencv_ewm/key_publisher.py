import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32
from geometry_msgs.msg import Point  # 确保导入 Point 消息类型
# import msvcrt  # Windows系统可使用 msqcrt，其他系统请使用适当的键盘输入方法

global key, center_x, center_y, depth_value, delta_x   #全局变量

key = "k"
center_x = 0
center_y = 0
depth_value = 0.0
delta_x = 0


class KeyPublisher(Node):
    def __init__(self):
        super().__init__('key_publisher')
        self.publisher_ = self.create_publisher(String, 'key_input', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 每0.1秒发布一次

        # 创建订阅者，订阅深度值
        self.subscription = self.create_subscription(
            Float32,
            'depth_value',
            self.depth_callback,
            10)  # QoS参数设置
        
        # 创建订阅器以订阅 red_center_point 主题
        self.subscription = self.create_subscription(
            Point, 
            'red_center_point', 
            self.center_callback, 
            10  # QoS参数设置
        )
        self.get_logger().info("Key Publisher started")
    
    def depth_callback(self, msg):
        global depth_value  # 声明使用全局变量
        if msg:  # 检查消息是否有效
            depth_value = int(msg.data)  # 将消息数据转换为整型
        # 打印接收到的深度值
        # self.get_logger().info(f'Received depth value: {depth_value} mm')

    def center_callback(self, msg):
        global center_x, center_y, delta_x  # 声明使用全局变量
        if msg:  # 检查消息是否有效
            center_x = msg.x
            center_y = msg.y
            delta_x = msg.z
        else:  # 如果未接收到有效数据，则设为零
            center_x = 0
            center_y = 0
            delta_x = 0 
        # self.get_logger().info(f'Received center point: ({center_x}, {center_y},{delta_x})')

    def timer_callback(self):
        global key, depth_value, delta_x,center_x  # 声明使用全局变量
        if(center_x > 0):
            if(depth_value > 0):
                key = "i"
                if(delta_x > 50.0):
                    key = "l"
                if(delta_x < -50.0):
                    key = "j"
            elif(depth_value == 0):
                if(delta_x > 50.0):
                    key = "l"
                elif(delta_x < -50.0):
                    key = "j"
                else:
                    key = "k"
            else:
                key = "k"
        else:
            key = "k"

        msg = String()
        msg.data = key
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{key}" {depth_value} mm ({center_x}, {center_y},{delta_x})')

def main(args=None):
    rclpy.init(args=args)
    key_publisher = KeyPublisher()
    rclpy.spin(key_publisher)
    key_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
