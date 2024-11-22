#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class DepthListener : public rclcpp::Node {
public:
    DepthListener(
        const std::string & image_topic = "/camera/depth/image_raw",
        const std::string & center_topic = "red_center_point",
        const std::string & depth_topic = "depth_value") 
    : Node("depth_listener") {
        // 创建一个QoS配置文件
        rclcpp::QoS qos(10);
        qos.keep_last(10);
        qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        qos.durability(rclcpp::DurabilityPolicy::Volatile);
        
        // 订阅深度图像
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic, qos,
            std::bind(&DepthListener::listener_callback, this, std::placeholders::_1));

        // 订阅中心点
        center_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            center_topic, qos,
            std::bind(&DepthListener::center_callback, this, std::placeholders::_1));

        // 创建深度值发布者
        depth_publisher_ = this->create_publisher<std_msgs::msg::Float32>(depth_topic, qos);
    }

private:
    // 深度图像回调函数
    void listener_callback(const sensor_msgs::msg::Image::SharedPtr msg) const {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            // 将ROS消息转换为OpenCV格式
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        } catch (const cv_bridge::Exception & e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // 检查是否接收到中心点的坐标
        if (!received_center_) {
            RCLCPP_WARN(this->get_logger(), "Center point not received yet.");
            return;
        }

        // 使用接收到的中心点坐标
        int center_x = static_cast<int>(center_point_.x);
        int center_y = static_cast<int>(center_point_.y);

        // 确保中心点坐标在图像范围内
        if (center_x < 0 || center_x >= cv_ptr->image.cols || center_y < 0 || center_y >= cv_ptr->image.rows) {
            RCLCPP_WARN(this->get_logger(), "Center point is out of image bounds: (%d, %d)", center_x, center_y);
            return;
        }

        // 获取中心点的深度值（注意：深度图像通常是16位）
        uint16_t depth_value = cv_ptr->image.at<uint16_t>(center_y, center_x);
        RCLCPP_INFO(this->get_logger(), "Depth at center: %d mm", depth_value);

        // 发布深度值
        auto depth_msg = std_msgs::msg::Float32();
        depth_msg.data = static_cast<float>(depth_value);
        depth_publisher_->publish(depth_msg);
    }

    void center_callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        center_point_ = *msg;
        received_center_ = true;
        //RCLCPP_INFO(this->get_logger(), "Received center point: (%.2f, %.2f)", center_point_.x, center_point_.y);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr center_subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr depth_publisher_;
    geometry_msgs::msg::Point center_point_;
    bool received_center_ = false;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthListener>());
    rclcpp::shutdown();
    return 0;
}
