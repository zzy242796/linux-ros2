import cv2
import rclpy
from rclpy.node import Node
# if run this python script should remove .
# it is correct to add . when run in ros2
from .utils import Colors, yaml_load,preprocessing_img,do_NMS,scale_boxes,box_label
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge 
import time
from rclpy.qos import QoSProfile

class Inference_node(Node):
    def __init__(self,node_name,model):
        super().__init__(node_name)
        self.model = model
        self.iou_thres = 0.25  # TF.js NMS: IoU threshold
        self.conf_thres = 0.45
        self.new_shape = (640,640)
        # gen color
        self.colors = Colors()
        # load names
        data = "/home/hngxy/temp/zhijing.yaml"
        self.names = yaml_load(data)["names"]
        qos = QoSProfile(depth=10)
        self.mat = None

        self.cv_bridge = CvBridge()
        self.get_logger().info(f"start reader:{node_name}")
        self.result_pub = self.create_publisher(Image,'result',10)
        self.center_pub = self.create_publisher(Point, 'red_center_point', 10)
        self.img_sub = self.create_subscription(Image,'/camera/color/image_raw',self._callback,qos)

    def _callback(self,msg):
        cv_img = self.cv_bridge.imgmsg_to_cv2(msg,'bgr8')
        input = preprocessing_img(cv_img,new_shape=self.new_shape)
        t = time.time()
        self.model.setInput(input)
        y = self.model.forward()
        # self.get_logger().info(f'inference time:{time.time()-t} s')
        
        pred = do_NMS(y,conf_thres=self.conf_thres,iou_thres=self.iou_thres)
        for i, det in enumerate(pred):
            #print(len(det))
            det[:, :4] = scale_boxes(input.shape[2:], det[:, :4], cv_img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                # print(c,conf)
                # 计算中心点
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                # 打印中心点坐标
                self.get_logger().info(f'Center point of box {i}: ({x_center}, {y_center})')
            
                box_label(xyxy, im=cv_img, color=self.colors(c, True), label=self.names[c])

                # 绘制中心点
                cv2.circle(cv_img, (x_center, y_center), 5, (0, 255, 0), -1)
            # cv2.imshow('result', cv_img)
            # cv2.waitKey(20)
            # cv2.destroyAllWindows()
            # print(pred.shape)
        result = self.cv_bridge.cv2_to_imgmsg(cv_img,'bgr8')
        self.result_pub.publish(result)
        center_msg = Point()
        center_msg.x = float(x_center)
        center_msg.y = float(y_center)
        self.center_pub.publish(center_msg)
        
def main(args=None): 
    model = cv2.dnn.readNetFromONNX('/home/hngxy/temp/best.onnx')

    rclpy.init(args=args)
    node = Inference_node(node_name="yolov5_inf_node",model = model)  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
