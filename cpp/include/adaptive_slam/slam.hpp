#pragma once

#include "adaptive_slam/controller.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace adaptive_slam {

enum class TrackingMode { LowCost, Recovery };

const char* modeName(TrackingMode mode) noexcept;

class AdaptiveController {
public:
    AdaptiveController(double enterRecoveryBelow = 0.48,
                       double exitRecoveryAbove = 0.68);
    TrackingMode choose(double confidence, TrackingMode currentMode) const;

private:
    adaptive_slam_controller controller_{};
};

struct CameraModel {
    double fx{};
    double fy{};
    double cx{};
    double cy{};
    cv::Matx33d matrix() const;
};

struct FrameResult {
    std::int64_t timestampNs{};
    cv::Vec3d position{0.0, 0.0, 0.0};
    double confidence{};
    double trackedRatio{};
    double inlierRatio{};
    double reprojectionErrorPx{std::numeric_limits<double>::infinity()};
    std::size_t activeFeatures{};
    int inliers{};
    double elapsedMs{};
    TrackingMode mode{TrackingMode::LowCost};
    int mapInliers{};
};

struct Keyframe {
    std::int64_t timestampNs{};
    cv::Matx33d rotationWorldCamera{cv::Matx33d::eye()};
    cv::Vec3d positionWorld{0.0, 0.0, 0.0};
};

class FeatureSLAM {
public:
    explicit FeatureSLAM(CameraModel camera, int minInliers = 12,
                         int keyframeInterval = 10);
    FrameResult process(const cv::Mat& image, std::int64_t timestampNs,
                        TrackingMode mode);

    const std::vector<cv::Point3d>& landmarks() const noexcept;
    const std::vector<Keyframe>& keyframes() const noexcept;

private:
    struct PoseEstimate {
        double inlierRatio{};
        double medianError{};
        int inlierCount{};
    };

    cv::Mat prepare(const cv::Mat& image, TrackingMode mode) const;
    std::vector<cv::Point2f> detect(const cv::Mat& image,
                                    TrackingMode mode) const;
    bool estimateMotion(const std::vector<cv::Point2f>& oldPoints,
                        const std::vector<cv::Point2f>& newPoints,
                        double& inlierRatio, double& reprojectionError,
                        int& inlierCount);
    std::optional<PoseEstimate> estimatePoseFromMap(
        const std::vector<cv::Point2f>& imagePoints,
        const std::vector<int>& landmarkIds);
    void triangulateNewLandmarks(
        const std::vector<cv::Point2f>& oldPoints,
        const std::vector<cv::Point2f>& newPoints,
        std::vector<int>& landmarkIds,
        const cv::Matx33d& previousRotation,
        const cv::Vec3d& previousPosition);
    static double confidenceScore(double trackedRatio, double inlierRatio,
                                  double reprojectionErrorPx);

    CameraModel camera_;
    int minInliers_;
    int keyframeInterval_;
    cv::Mat previousRawImage_;
    std::vector<cv::Point2f> previousPoints_;
    std::vector<int> previousLandmarkIds_;
    cv::Matx33d rotationWorldCamera_{cv::Matx33d::eye()};
    cv::Vec3d positionWorld_{0.0, 0.0, 0.0};
    std::vector<cv::Point3d> landmarks_;
    std::vector<Keyframe> keyframes_;
    int frameIndex_{};
};

struct EuRoCFrame {
    std::int64_t timestampNs{};
    std::filesystem::path imagePath;
};

struct GroundTruth {
    std::vector<std::int64_t> timestampsNs;
    std::vector<cv::Vec3d> positions;
};

struct EuRoCSequence {
    CameraModel camera;
    std::vector<EuRoCFrame> frames;
    std::optional<GroundTruth> groundTruth;
};

EuRoCSequence loadEuRoC(const std::filesystem::path& sequencePath);
cv::Mat applyDegradation(const cv::Mat& image, const std::string& degradation,
                         std::uint32_t seed);

struct RunSummary {
    std::string sequence;
    std::string configuration;
    std::string degradation;
    std::size_t frames{};
    std::size_t landmarks{};
    std::size_t keyframes{};
    std::size_t framesWithMapPose{};
    double ateRmse{std::numeric_limits<double>::quiet_NaN()};
    std::size_t trackingFailures{};
    double failureRate{};
    double meanLatencyMs{};
    double p95LatencyMs{};
    double wallTimeSeconds{};
    double cpuTimeSeconds{};
    std::int64_t rssDeltaBytes{};
    std::int64_t rssAfterBytes{};
    double recoveryFraction{};
    std::size_t modeTransitions{};
};

RunSummary runConfiguration(const std::filesystem::path& sequencePath,
                            const std::filesystem::path& outputPath,
                            const std::string& configuration,
                            const std::string& degradation,
                            std::uint32_t seed = 17);
std::vector<RunSummary> runBenchmark(
    const std::filesystem::path& sequencePath,
    const std::filesystem::path& outputPath,
    const std::vector<std::string>& degradations);
void writeSummaries(const std::filesystem::path& outputPath,
                    const std::vector<RunSummary>& summaries);

}  // namespace adaptive_slam

#include "adaptive_slam/uav_hybrid_slam.hpp"
