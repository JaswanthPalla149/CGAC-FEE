#ifndef ADAPTIVE_SLAM_UAV_HYBRID_SLAM_HPP
#define ADAPTIVE_SLAM_UAV_HYBRID_SLAM_HPP

#include "adaptive_slam/controller.h"
#include "adaptive_slam/hbst_matcher.hpp"
#include "adaptive_slam/low_mac_features.hpp"
#include "adaptive_slam/slam.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>
#include <optional>
#include <string>

namespace adaptive_slam {

enum class HybridPowerMode {
    LowCost = 0,         // Nominal tracking: 450 features, 2 pyramid levels, 5x LiDAR stride
    Recovery = 1,        // Perceptual recovery: Integer CLAHE, 1200 features, 4 levels, 1x LiDAR
    VisualBypassLIO = 2  // Extreme lighting shock (0 Lux darkness / solar glare): 0 visual MACs, 100% LIO
};

const char* hybridModeName(HybridPowerMode mode) noexcept;

struct ImuMeasurement {
    std::int64_t timestampNs{0};
    cv::Vec3d angularVel{0.0, 0.0, 0.0};  // rad/s (gyroscope)
    cv::Vec3d linearAccel{0.0, 0.0, 0.0}; // m/s^2 specific force in body frame
};

struct LidarPoint {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float intensity{0.0f};
};

struct LidarScan {
    std::int64_t timestampNs{0};
    std::vector<LidarPoint> points;
};

struct UAVState {
    std::int64_t timestampNs{0};
    cv::Matx33d rotationWorldBody{cv::Matx33d::eye()};
    cv::Vec3d positionWorld{0.0, 0.0, 0.0};       // Metric scale (meters)
    cv::Vec3d velocityWorld{0.0, 0.0, 0.0};       // Metric scale (m/s)
    cv::Vec3d gyroBias{0.0, 0.0, 0.0};
    cv::Vec3d accelBias{0.0, 0.0, 0.0};
    double confidence{1.0};
    HybridPowerMode mode{HybridPowerMode::LowCost};
    int visualInliers{0};
    int lidarInliers{0};
    double elapsedMs{0.0};
    std::uint64_t estimatedMacCount{0};           // Hardware MAC count for power comparison
    double estimatedEnergyJoules{0.0};
};

struct HybridSlamConfig {
    double enterRecoveryBelow{0.48};
    double visualBypassBelow{0.20};
    double exitRecoveryAbove{0.68};
    cv::Vec3d gravity{0.0, 0.0, -9.81};          // Metric gravity vector
    int minInliers{12};
    bool enableIntegerClahe{true};
    bool enableHbstMatching{true};
};

class MultiSensorController {
public:
    explicit MultiSensorController(double enterRecoveryBelow = 0.48,
                                   double visualBypassBelow = 0.20,
                                   double exitRecoveryAbove = 0.68);

    HybridPowerMode choose(double visualConfidence,
                           double lidarDegeneracy,
                           HybridPowerMode currentMode) const;

private:
    double enterRecoveryBelow_;
    double visualBypassBelow_;
    double exitRecoveryAbove_;
};

class UAVHybridSLAM {
public:
    explicit UAVHybridSLAM(CameraModel camera, const HybridSlamConfig& config = {});

    // Integrate one timestamped body-frame gyro and accelerometer sample.
    void processImu(const ImuMeasurement& imu);

    // Synchronized multi-modal frame processing
    // If image is pitch-black or blinded, automatically shifts to VisualBypassLIO (0 visual MACs)
    UAVState processFrame(const cv::Mat& image,
                          std::int64_t timestampNs,
                          const std::optional<LidarScan>& scan = std::nullopt);

    const UAVState& currentState() const noexcept { return state_; }
    const std::vector<UAVState>& trajectory() const noexcept { return trajectory_; }
    std::uint64_t totalMacOperations() const noexcept { return totalMacs_; }
    double totalEnergyJoules() const noexcept { return totalJoules_; }

    const std::vector<cv::Point3d>& globalLidarMap() const noexcept { return globalLidarMap_; }
    const std::vector<cv::Point3d>& currentScanWorld() const noexcept { return currentScanWorld_; }
    const std::vector<cv::Point2f>& currentTrackedPoints() const noexcept { return previousPoints_; }
    const cv::Mat& currentProcessedImage() const noexcept { return previousRawImage_; }

    void reset(const cv::Vec3d& initialPosition = {0.0, 0.0, 0.0});

private:
    CameraModel camera_;
    HybridSlamConfig config_;
    MultiSensorController controller_;
    UAVState state_{};
    std::vector<UAVState> trajectory_;

    std::int64_t lastImuTimestampNs_{0};
    bool hasImuTimestamp_{};
    cv::Mat previousRawImage_;
    std::vector<cv::Point2f> previousPoints_;
    std::vector<int> previousLandmarkIds_;
    std::vector<cv::Point3d> metricLandmarks_;
    std::vector<BinaryDescriptor> landmarkDescriptors_;
    HBSTTree landmarkIndex_;

    std::vector<cv::Point3d> globalLidarMap_;
    std::vector<cv::Point3d> currentScanWorld_;

    std::uint64_t totalMacs_{0};
    double totalJoules_{0.0};

    // Internal low-MAC subroutines
    cv::Mat prepareImage(const cv::Mat& image, HybridPowerMode mode) const;
    std::vector<cv::Point2f> detectCorners(const cv::Mat& image, HybridPowerMode mode) const;
    void updateLidarPoints(const LidarScan& scan, HybridPowerMode mode, int& lidarInliers);
    bool alignLidarScan(const LidarScan& scan, cv::Matx33d& deltaR, cv::Vec3d& deltaT, int& inliers, HybridPowerMode mode);
    void updateMap(const LidarScan& scan, const cv::Matx33d& R, const cv::Vec3d& t);
};

// Synthetic generator for testing sudden illumination collapse on UAVs (e.g. tunnel entry, solar flare)
cv::Mat applyIlluminationShock(const cv::Mat& input, const std::string& shockType);

} // namespace adaptive_slam

#endif // ADAPTIVE_SLAM_UAV_HYBRID_SLAM_HPP
