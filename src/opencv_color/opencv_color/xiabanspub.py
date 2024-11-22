import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile

class LowerBodyDetector(Node):
    def __init__(self):
        super().__init__('lower_body_detector')
        self.publisher_image = self.create_publisher(Image, 'xiabans_image', 10)
        self.publisher_point = self.create_publisher(Point, 'red_center_point', 10)
        self.bridge = CvBridge()
        qos = QoSProfile(depth=10)
        self.lower_body_cascade = cv2.CascadeClassifier('src/ziyuan/haarcascade_lowerbody.xml')
        
        # 创建订阅器
        self.subscription = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            qos
        )

        self.timer = self.create_timer(0.05, self.detect_lower_body)

        # 存储最新的图像帧
        self.current_frame = None

    def image_callback(self, msg):
        # 将ROS图像消息转换为OpenCV图像
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def detect_lower_body(self):
        # 确保存在有效的图像帧
        if self.current_frame is None:
            self.get_logger().error("没有接收到图像帧")
            return
        
        # 初始化中心点坐标
        center_x = 0.0
        center_y = 0.0
        delta_x = 0.0

        frame = self.current_frame

        # 将图像转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用分类器检测下半身
        lower_bodies = self.lower_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(lower_bodies) > 0:
            # 找到面积最大的矩形框
            max_rect = max(lower_bodies, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = max_rect

            # 在最大矩形框周围绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 计算中心点并绘制
            center_x = float(x + w // 2)
            center_y = float(y + h // 2)

            # 获取图像的宽高
            height, width, _ = frame.shape
            image_center_x = float(width // 2)  # 图像中心横坐标

            # 计算中心点差值
            delta_x = center_x - image_center_x
            
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)  # 绿色圆点

        # 发布中心点坐标
        center_point = Point()
        center_point.x = center_x
        center_point.y = center_y
        center_point.z = delta_x  
        self.publisher_point.publish(center_point)
        self.get_logger().info(f"Published center point: ({int(center_x)}, {int(center_y)},{delta_x})")

        # 显示结果
        cv2.imshow('Lower Body Detection', frame)
        cv2.waitKey(1)

        # 发布图像
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_image.publish(image_msg)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    lower_body_detector = LowerBodyDetector()
    try:
        rclpy.spin(lower_body_detector)
    except KeyboardInterrupt:
        pass
    finally:
        lower_body_detector.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
