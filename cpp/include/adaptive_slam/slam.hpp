#pragma once

#include "adaptive_slam/controller.h"
#include "adaptive_slam/hbst_matcher.hpp"
#include "adaptive_slam/pose_graph.hpp"

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

inline constexpr double kDefaultEnterRecoveryBelow = 0.48;
inline constexpr double kDefaultExitRecoveryAbove = 0.60;
inline constexpr double kDefaultEnterVisualInertialBelow = 0.40;
inline constexpr double kDefaultExitVisualInertialAbove = 0.48;

enum class TrackingMode { LowCost, Recovery, VisualInertial };

const char* modeName(TrackingMode mode) noexcept;

class AdaptiveController {
public:
    AdaptiveController(double enterRecoveryBelow = kDefaultEnterRecoveryBelow,
                       double exitRecoveryAbove = kDefaultExitRecoveryAbove,
                       double enterVisualInertialBelow =
                           kDefaultEnterVisualInertialBelow,
                       double exitVisualInertialAbove =
                           kDefaultExitVisualInertialAbove);
    TrackingMode choose(double confidence, TrackingMode currentMode,
                        bool imuAvailable = false) const;

private:
    adaptive_slam_controller controller_{};
    double enterVisualInertialBelow_{};
    double exitVisualInertialAbove_{};
};

struct CameraModel {
    double fx{};
    double fy{};
    double cx{};
    double cy{};
    cv::Vec4d distortion{};
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
    bool trackingSucceeded{};
    bool inertialFallback{};
    bool loopClosed{false};
    int loopCandidateId{-1};
    double loopDriftEstimateUnits{0.0};
};

struct Keyframe {
    std::size_t id{0};
    std::int64_t timestampNs{0};
    cv::Matx33d rotationWorldCamera{cv::Matx33d::eye()};
    cv::Vec3d positionWorld{0.0, 0.0, 0.0};
    std::vector<cv::Point2f> points;
    std::vector<int> landmarkIds;
    std::vector<BinaryDescriptor> descriptors;

    Keyframe() = default;
    Keyframe(std::int64_t ts, const cv::Matx33d& R, const cv::Vec3d& t, std::size_t kf_id = 0)
        : id(kf_id), timestampNs(ts), rotationWorldCamera(R), positionWorld(t) {}
};

class FeatureSLAM {
public:
    explicit FeatureSLAM(CameraModel camera, int minInliers = 12,
                         int keyframeInterval = 10);
    FrameResult process(const cv::Mat& image, std::int64_t timestampNs,
                        TrackingMode mode,
                        std::optional<cv::Matx33d> gyroRotationPrior =
                            std::nullopt,
                        std::optional<cv::Vec3d> inertialTranslation =
                            std::nullopt);

    const std::vector<cv::Point3d>& landmarks() const noexcept;
    const std::vector<Keyframe>& keyframes() const noexcept;
    const std::vector<cv::Point2f>& currentFeatures() const noexcept {
        return previousPoints_;
    }
    const cv::Matx33d& rotationWorldCamera() const noexcept {
        return rotationWorldCamera_;
    }
    std::size_t loopClosures() const noexcept { return loopClosures_; }
    double maxLoopDriftEstimateUnits() const noexcept {
        return maxLoopDriftEstimateUnits_;
    }
    const std::vector<cv::Vec3d>& correctedFramePositions() const noexcept {
        return framePoseHistory_;
    }

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
                        int& inlierCount,
                        const std::optional<cv::Matx33d>& gyroRotationPrior);
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
    bool detectAndCloseLoop(Keyframe& currentKf, FrameResult& result);

    CameraModel camera_;
    int minInliers_;
    int keyframeInterval_;
    cv::Mat previousRawImage_;
    cv::Mat previousPreparedLowCost_;
    cv::Mat previousPreparedRecovery_;
    std::vector<cv::Point2f> previousPoints_;
    std::vector<int> previousLandmarkIds_;
    cv::Matx33d rotationWorldCamera_{cv::Matx33d::eye()};
    cv::Vec3d positionWorld_{0.0, 0.0, 0.0};
    std::vector<cv::Point3d> landmarks_;
    std::vector<std::size_t> landmarkAnchorKeyframes_;
    std::vector<Keyframe> keyframes_;
    std::vector<PoseGraphConstraint> poseGraphConstraints_;
    std::vector<cv::Vec3d> framePoseHistory_;
    std::vector<std::size_t> frameAnchorKeyframes_;
    int frameIndex_{};
    std::size_t loopClosures_{0};
    double maxLoopDriftEstimateUnits_{0.0};
    std::size_t lastLoopClosureKeyframe_{};
    bool hasLastLoopClosure_{false};
};

struct EuRoCFrame {
    std::int64_t timestampNs{};
    std::filesystem::path imagePath;
};

struct GroundTruth {
    std::vector<std::int64_t> timestampsNs;
    std::vector<cv::Vec3d> positions;
};

struct GyroscopeSample {
    std::int64_t timestampNs{};
    cv::Vec3d angularVelocity{};
    cv::Vec3d linearAcceleration{};
};

struct InertialPreintegration {
    cv::Vec3d displacement{};
    cv::Vec3d velocity{};
    cv::Matx33d rotationWorldCamera{cv::Matx33d::eye()};
};

struct EuRoCSequence {
    CameraModel camera;
    std::vector<EuRoCFrame> frames;
    std::optional<GroundTruth> groundTruth;
    std::vector<GyroscopeSample> gyroscope;
    cv::Matx33d cameraFromImu{cv::Matx33d::eye()};
};

EuRoCSequence loadEuRoC(const std::filesystem::path& sequencePath);
std::optional<cv::Matx33d> integrateGyroscopeRotation(
    const std::vector<GyroscopeSample>& samples,
    std::int64_t startTimestampNs, std::int64_t endTimestampNs,
    const cv::Matx33d& cameraFromImu);
std::optional<InertialPreintegration> integrateInertialTranslation(
    const std::vector<GyroscopeSample>& samples,
    std::int64_t startTimestampNs, std::int64_t endTimestampNs,
    const cv::Matx33d& rotationWorldCamera,
    const cv::Matx33d& cameraFromImu, const cv::Vec3d& gravityWorld,
    const cv::Vec3d& initialVelocity);
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
    std::size_t framesWithPose{};
    std::size_t groundTruthMatchedFrames{};
    double ateRmse{std::numeric_limits<double>::quiet_NaN()};
    double rpeTranslationRmse{std::numeric_limits<double>::quiet_NaN()};
    double poseCoverage{std::numeric_limits<double>::quiet_NaN()};
    std::string trajectoryStatus{"no_ground_truth"};
    std::size_t trackingFailures{};
    std::size_t inertialFallbackFrames{};
    double failureRate{};
    double meanLatencyMs{};
    double p95LatencyMs{};
    double wallTimeSeconds{};
    double cpuTimeSeconds{};
    std::int64_t rssDeltaBytes{};
    std::int64_t rssAfterBytes{};
    std::int64_t peakRssBytes{};
    double energyJoules{std::numeric_limits<double>::quiet_NaN()};
    std::string energyMethod{"not_measured"};
    std::string energyScope{"not_available"};
    std::string energyStatus{"unavailable"};
    double controllerEnterThreshold{kDefaultEnterRecoveryBelow};
    double controllerExitThreshold{kDefaultExitRecoveryAbove};
    double controllerEnterVisualInertialThreshold{
        kDefaultEnterVisualInertialBelow};
    double controllerExitVisualInertialThreshold{
        kDefaultExitVisualInertialAbove};
    double recoveryFraction{};
    double visualInertialFraction{};
    std::size_t modeTransitions{};
    std::size_t loopClosures{};
    double maxLoopDriftEstimateUnits{};
};

RunSummary runConfiguration(const std::filesystem::path& sequencePath,
                            const std::filesystem::path& outputPath,
                            const std::string& configuration,
                            const std::string& degradation,
                            std::uint32_t seed = 17,
                            double enterRecoveryBelow = kDefaultEnterRecoveryBelow,
                            double exitRecoveryAbove = kDefaultExitRecoveryAbove,
                            double enterVisualInertialBelow =
                                kDefaultEnterVisualInertialBelow,
                            double exitVisualInertialAbove =
                                kDefaultExitVisualInertialAbove,
                            std::size_t maxFrames = 0,
                            bool visualize = false);
std::vector<RunSummary> runBenchmark(
    const std::filesystem::path& sequencePath,
    const std::filesystem::path& outputPath,
    const std::vector<std::string>& degradations,
    double enterRecoveryBelow = kDefaultEnterRecoveryBelow,
    double exitRecoveryAbove = kDefaultExitRecoveryAbove,
    double enterVisualInertialBelow = kDefaultEnterVisualInertialBelow,
    double exitVisualInertialAbove = kDefaultExitVisualInertialAbove,
    const std::vector<std::string>& configurations = {
        "fixed_low", "fixed_high", "adaptive"},
    std::size_t maxFrames = 0, bool visualize = false);
void writeSummaries(const std::filesystem::path& outputPath,
                    const std::vector<RunSummary>& summaries);

}  // namespace adaptive_slam

#include "adaptive_slam/uav_hybrid_slam.hpp"
