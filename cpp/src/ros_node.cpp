#include "adaptive_slam/uav_hybrid_slam.hpp"
#include "livox_ros_driver/CustomMsg.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

class CGACSlamNode {
public:
    CGACSlamNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
        double fx, fy, cx, cy;
        pnh.param<double>("fx", fx, 646.78);
        pnh.param<double>("fy", fy, 646.66);
        pnh.param<double>("cx", cx, 313.46);
        pnh.param<double>("cy", cy, 261.40);

        std::string cameraTopic, imuTopic, lidarTopic;
        pnh.param<std::string>("camera_topic", cameraTopic, "/left_camera/image");
        pnh.param<std::string>("imu_topic", imuTopic, "/livox/imu");
        pnh.param<std::string>("lidar_topic", lidarTopic, "/livox/lidar");

        adaptive_slam::CameraModel camera{fx, fy, cx, cy};
        adaptive_slam::HybridSlamConfig config;
        config.enableIntegerClahe = true;
        config.enableHbstMatching = true;
        slam_ = std::make_unique<adaptive_slam::UAVHybridSLAM>(camera, config);

        odomPub_ = nh.advertise<nav_msgs::Odometry>("/cgac_slam/odometry", 10);
        pathPub_ = nh.advertise<nav_msgs::Path>("/cgac_slam/path", 10);

        imuSub_ = nh.subscribe(imuTopic, 1000, &CGACSlamNode::imuCallback, this);
        lidarSub_ = nh.subscribe(lidarTopic, 20, &CGACSlamNode::lidarCallback, this);
        imageSub_ = nh.subscribe(cameraTopic, 10, &CGACSlamNode::imageCallback, this);

        ROS_INFO_STREAM("=================================================");
        ROS_INFO_STREAM(" CGAC-SLAM (EcoLIVO) Live Simulation Node Ready  ");
        ROS_INFO_STREAM("=================================================");
        ROS_INFO_STREAM(" Camera Sub: " << cameraTopic);
        ROS_INFO_STREAM(" IMU Sub   : " << imuTopic);
        ROS_INFO_STREAM(" LiDAR Sub : " << lidarTopic);
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        adaptive_slam::ImuMeasurement imu;
        imu.timestampNs = msg->header.stamp.toNSec();
        imu.angularVel = {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
        imu.linearAccel = {msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z};

        std::lock_guard<std::mutex> lock(slamMutex_);
        slam_->processImu(imu);
    }

    void lidarCallback(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
        adaptive_slam::LidarScan scan;
        scan.timestampNs = msg->header.stamp.toNSec();
        scan.points.reserve(msg->point_num);
        for (std::size_t i = 0; i < msg->point_num; ++i) {
            const auto& pt = msg->points[i];
            scan.points.push_back({pt.x, pt.y, pt.z, static_cast<float>(pt.reflectivity)});
        }
        std::lock_guard<std::mutex> lock(scanMutex_);
        latestScan_ = std::move(scan);
    }

    void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        } catch (const cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        std::optional<adaptive_slam::LidarScan> scan;
        {
            std::lock_guard<std::mutex> lock(scanMutex_);
            if (latestScan_.has_value()) {
                scan = latestScan_;
                latestScan_.reset();
            }
        }

        adaptive_slam::UAVState state;
        {
            std::lock_guard<std::mutex> lock(slamMutex_);
            state = slam_->processFrame(cv_ptr->image, msg->header.stamp.toNSec(), scan);
        }

        frameCount_++;
        if (frameCount_ % 20 == 0 || state.mode == adaptive_slam::HybridPowerMode::Recovery) {
            ROS_INFO("[CGAC-SLAM] Frame #%5d | Mode: %-15s | Conf: %4.2f | MACs: %10lu | Energy: %5.3f J",
                     frameCount_, adaptive_slam::hybridModeName(state.mode),
                     state.confidence, slam_->totalMacOperations(), slam_->totalEnergyJoules());
        }

        publishOdometry(msg->header.stamp, state);
    }

private:
    void publishOdometry(const ros::Time& stamp, const adaptive_slam::UAVState& state) {
        nav_msgs::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "world";
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = state.positionWorld[0];
        odom.pose.pose.position.y = state.positionWorld[1];
        odom.pose.pose.position.z = state.positionWorld[2];

        // Convert rotation to quaternion
        const auto& R = state.rotationWorldBody;
        double tr = R(0, 0) + R(1, 1) + R(2, 2);
        double qw = 1.0, qx = 0.0, qy = 0.0, qz = 0.0;
        if (tr > 0.0) {
            double S = std::sqrt(tr + 1.0) * 2.0;
            qw = 0.25 * S;
            qx = (R(2, 1) - R(1, 2)) / S;
            qy = (R(0, 2) - R(2, 0)) / S;
            qz = (R(1, 0) - R(0, 1)) / S;
        } else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2))) {
            double S = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
            qw = (R(2, 1) - R(1, 2)) / S;
            qx = 0.25 * S;
            qy = (R(0, 1) + R(1, 0)) / S;
            qz = (R(0, 2) + R(2, 0)) / S;
        } else if (R(1, 1) > R(2, 2)) {
            double S = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
            qw = (R(0, 2) - R(2, 0)) / S;
            qx = (R(0, 1) + R(1, 0)) / S;
            qy = 0.25 * S;
            qz = (R(1, 2) + R(2, 1)) / S;
        } else {
            double S = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
            qw = (R(1, 0) - R(0, 1)) / S;
            qx = (R(0, 2) + R(2, 0)) / S;
            qy = (R(1, 2) + R(2, 1)) / S;
            qz = 0.25 * S;
        }

        odom.pose.pose.orientation.w = qw;
        odom.pose.pose.orientation.x = qx;
        odom.pose.pose.orientation.y = qy;
        odom.pose.pose.orientation.z = qz;
        odomPub_.publish(odom);

        geometry_msgs::PoseStamped ps;
        ps.header = odom.header;
        ps.pose = odom.pose.pose;
        path_.header = odom.header;
        path_.poses.push_back(ps);
        pathPub_.publish(path_);

        tf::Transform transform;
        transform.setOrigin(tf::Vector3(state.positionWorld[0], state.positionWorld[1], state.positionWorld[2]));
        tf::Quaternion q(qx, qy, qz, qw);
        transform.setRotation(q);
        tfBroadcaster_.sendTransform(tf::StampedTransform(transform, stamp, "world", "base_link"));
    }

    std::unique_ptr<adaptive_slam::UAVHybridSLAM> slam_;
    std::mutex slamMutex_;
    std::mutex scanMutex_;
    std::optional<adaptive_slam::LidarScan> latestScan_;

    ros::Subscriber imuSub_;
    ros::Subscriber lidarSub_;
    ros::Subscriber imageSub_;
    ros::Publisher odomPub_;
    ros::Publisher pathPub_;
    tf::TransformBroadcaster tfBroadcaster_;
    nav_msgs::Path path_;
    int frameCount_{0};
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cgac_mapping");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    CGACSlamNode node(nh, pnh);
    ros::spin();
    return 0;
}
