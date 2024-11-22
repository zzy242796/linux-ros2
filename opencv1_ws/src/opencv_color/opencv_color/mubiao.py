import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class JsonPointPublisher(Node):
    def __init__(self):
        super().__init__('json_point_publisher')
        self.publisher_ = self.create_publisher(Point, 'red_center_point', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 1秒发布一次
        self.file_path = '/home/hngxy/Downloads/cc.json'

    def read_value_from_json(self, point):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                return data.get(point, None)
        except Exception as e:
            self.get_logger().error(f"读取 JSON 文件时出错: {e}")
            return None

    def timer_callback(self):
        center_x = self.read_value_from_json('center_x')
        center_y = self.read_value_from_json('center_y')

        if center_x is not None and center_y is not None:
            center_msg = Point()
            center_msg.x = float(center_x)
            center_msg.y = float(center_y)
            # msg.point.z = 0.0  # 假设 z 轴为 0

            self.publisher_.publish(center_msg)
            self.get_logger().info(f'发布坐标点: ({center_x}, {center_y})')

def main(args=None):
    rclpy.init(args=args)
    point_publisher = JsonPointPublisher()
    rclpy.spin(point_publisher)

    point_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

