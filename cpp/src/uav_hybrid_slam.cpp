#include "adaptive_slam/uav_hybrid_slam.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>

namespace adaptive_slam {

const char* hybridModeName(HybridPowerMode mode) noexcept {
    switch (mode) {
        case HybridPowerMode::LowCost:
            return "low_cost";
        case HybridPowerMode::Recovery:
            return "recovery";
        case HybridPowerMode::VisualBypassLIO:
            return "visual_bypass_lio";
    }
    return "unknown";
}

MultiSensorController::MultiSensorController(double enterRecoveryBelow,
                                             double visualBypassBelow,
                                             double exitRecoveryAbove)
    : enterRecoveryBelow_(enterRecoveryBelow),
      visualBypassBelow_(visualBypassBelow),
      exitRecoveryAbove_(exitRecoveryAbove) {}

HybridPowerMode MultiSensorController::choose(double visualConfidence,
                                               double lidarDegeneracy,
                                               HybridPowerMode currentMode) const {
    // 1. Extreme optical failure (Pitch darkness / blinding flare)
    if (visualConfidence < visualBypassBelow_) {
        return HybridPowerMode::VisualBypassLIO;
    }

    // 2. Recovery / LowCost state machine with hysteresis
    if (currentMode == HybridPowerMode::VisualBypassLIO) {
        if (visualConfidence > exitRecoveryAbove_) {
            return HybridPowerMode::LowCost;
        }
        if (visualConfidence > enterRecoveryBelow_) {
            return HybridPowerMode::Recovery;
        }
        return HybridPowerMode::VisualBypassLIO;
    }

    if (currentMode == HybridPowerMode::LowCost) {
        if (visualConfidence < enterRecoveryBelow_ || lidarDegeneracy > 0.85) {
            return HybridPowerMode::Recovery;
        }
        return HybridPowerMode::LowCost;
    }

    // Current is Recovery
    if (visualConfidence > exitRecoveryAbove_ && lidarDegeneracy < 0.60) {
        return HybridPowerMode::LowCost;
    }
    return HybridPowerMode::Recovery;
}

UAVHybridSLAM::UAVHybridSLAM(CameraModel camera, const HybridSlamConfig& config)
    : camera_(camera),
      config_(config),
      controller_(config.enterRecoveryBelow, config.visualBypassBelow, config.exitRecoveryAbove) {
    reset();
}

void UAVHybridSLAM::reset(const cv::Vec3d& initialPosition) {
    state_ = UAVState{};
    state_.positionWorld = initialPosition;
    state_.rotationWorldBody = cv::Matx33d::eye();
    state_.confidence = 1.0;
    state_.mode = HybridPowerMode::LowCost;
    trajectory_.clear();
    previousPoints_.clear();
    previousLandmarkIds_.clear();
    metricLandmarks_.clear();
    landmarkDescriptors_.clear();
    previousRawImage_.release();
    lastImuTimestampNs_ = 0;
    totalMacs_ = 0;
    totalJoules_ = 0.0;
}

void UAVHybridSLAM::processImu(const ImuMeasurement& imu) {
    if (lastImuTimestampNs_ == 0) {
        lastImuTimestampNs_ = imu.timestampNs;
        return;
    }

    const double dt = std::max(1e-5, static_cast<double>(imu.timestampNs - lastImuTimestampNs_) * 1e-9);
    lastImuTimestampNs_ = imu.timestampNs;

    // Unbiased angular velocity and acceleration
    const cv::Vec3d unbOmega = imu.angularVel - state_.gyroBias;
    const cv::Vec3d unbAccel = imu.linearAccel - state_.accelBias;

    // World acceleration: a_W = R * a_body + gravity
    const cv::Vec3d accelWorld = state_.rotationWorldBody * unbAccel + config_.gravity;

    // Kinematic integration (Fixed-point friendly: 18 mults)
    state_.positionWorld += state_.velocityWorld * dt + 0.5 * accelWorld * dt * dt;
    state_.velocityWorld += accelWorld * dt;

    // Rodrigues small-angle rotation integration
    const double angle = cv::norm(unbOmega) * dt;
    if (angle > 1e-8) {
        const cv::Vec3d axis = unbOmega / (cv::norm(unbOmega) + 1e-12);
        cv::Mat rotMat;
        cv::Rodrigues(axis * angle, rotMat);
        cv::Matx33d dR;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                dR(r, c) = rotMat.at<double>(r, c);
            }
        }
        state_.rotationWorldBody = state_.rotationWorldBody * dR;
    }

    state_.timestampNs = imu.timestampNs;
    totalMacs_ += 45; // Approximately 45 MACs for high-rate IMU integration
}

cv::Mat UAVHybridSLAM::prepareImage(const cv::Mat& image, HybridPowerMode mode) const {
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }

    if (mode != HybridPowerMode::Recovery) {
        return gray;
    }

    if (config_.enableIntegerClahe) {
        // Integer-only CLAHE (0 floating point MACs!)
        return applyIntegerClahe(gray, 3, 8, 8);
    }
    return gray;
}

std::vector<cv::Point2f> UAVHybridSLAM::detectCorners(const cv::Mat& image, HybridPowerMode mode) const {
    const int maxCorners = (mode == HybridPowerMode::Recovery) ? 1200 : 450;
    const double quality = (mode == HybridPowerMode::Recovery) ? 0.008 : 0.015;
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(image, points, maxCorners, quality, 7.0, cv::noArray(), 7, false, 0.04);
    return points;
}

void UAVHybridSLAM::updateLidarPoints(const LidarScan& scan, HybridPowerMode mode, int& lidarInliers) {
    lidarInliers = 0;
    if (scan.points.empty()) return;

    // Dynamic stride based on power state (Downsamples in LowCost mode)
    const std::size_t stride = (mode == HybridPowerMode::LowCost) ? 5U : 1U;

    for (std::size_t i = 0; i < scan.points.size(); i += stride) {
        const auto& pt = scan.points[i];
        if (pt.z > 0.3f && pt.z < 60.0f) {
            ++lidarInliers;
            // Metric scale calibration: populate 3D spatial points
            if (metricLandmarks_.size() < 2048 && (i % 20 == 0)) {
                cv::Vec3d pBody(pt.x, pt.y, pt.z);
                cv::Vec3d pWorld = state_.positionWorld + state_.rotationWorldBody * pBody;
                metricLandmarks_.emplace_back(pWorld[0], pWorld[1], pWorld[2]);
            }
        }
    }
    totalMacs_ += (scan.points.size() / stride) * 3; // 3 MACs per point projection
}

UAVState UAVHybridSLAM::processFrame(const cv::Mat& image,
                                     std::int64_t timestampNs,
                                     const std::optional<LidarScan>& scan) {
    const auto startTime = std::chrono::steady_clock::now();
    state_.timestampNs = timestampNs;
    std::uint64_t frameMacs = 0;

    // 1. Fast Optical Invariant Test for Sudden Lighting Collapse
    cv::Scalar meanIntensity, stdDevIntensity;
    cv::meanStdDev(image, meanIntensity, stdDevIntensity);
    const double meanVal = meanIntensity[0];
    const double stdVal = stdDevIntensity[0];

    bool suddenLightCollapse = (meanVal < 6.0 || meanVal > 248.0 || stdVal < 3.0);
    double initialConfidence = suddenLightCollapse ? 0.05 :
        (state_.mode == HybridPowerMode::VisualBypassLIO ? 0.55 : state_.confidence);

    // 2. Controller Evaluation
    const double lidarDegeneracy = scan.has_value() && scan->points.size() < 30 ? 0.90 : 0.10;
    const HybridPowerMode chosenMode = controller_.choose(initialConfidence, lidarDegeneracy, state_.mode);
    state_.mode = chosenMode;

    int visualInliers = 0;
    int lidarInliers = 0;

    // 3. Optional LiDAR update
    if (scan.has_value()) {
        updateLidarPoints(*scan, chosenMode, lidarInliers);
    }

    // 4. Visual Execution OR Zero-MAC Visual Bypass
    if (chosenMode == HybridPowerMode::VisualBypassLIO) {
        // Sudden blackout or blinding flash: BYPASS visual pipeline completely!
        // 0 visual MACs spent. UAV continues on LiDAR + IMU integration!
        state_.confidence = 0.05;
        state_.visualInliers = 0;
        state_.lidarInliers = lidarInliers;
        previousPoints_.clear();
        previousRawImage_.release();
    } else {
        // Process visual stream with selected feature budget
        const cv::Mat processedImage = prepareImage(image, chosenMode);

        if (previousRawImage_.empty() || previousPoints_.size() < 10) {
            previousRawImage_ = processedImage.clone();
            previousPoints_ = detectCorners(processedImage, chosenMode);
            previousLandmarkIds_.assign(previousPoints_.size(), -1);
            frameMacs += previousPoints_.size() * 50;
            state_.confidence = previousPoints_.size() >= 10 ? 1.0 : 0.0;
            state_.visualInliers = static_cast<int>(previousPoints_.size());
            state_.lidarInliers = lidarInliers;
        } else {
            // Lucas-Kanade tracking
            std::vector<cv::Point2f> candidatePoints;
            std::vector<unsigned char> status;
            std::vector<float> errors;
            const cv::Size winSize = (chosenMode == HybridPowerMode::Recovery) ? cv::Size(31, 31) : cv::Size(21, 21);
            const int pyrLevels = (chosenMode == HybridPowerMode::Recovery) ? 4 : 2;

            cv::calcOpticalFlowPyrLK(previousRawImage_, processedImage, previousPoints_,
                                     candidatePoints, status, errors, winSize, pyrLevels);

            std::vector<cv::Point2f> trackedOld;
            std::vector<cv::Point2f> trackedNew;
            std::vector<int> trackedIds;
            for (std::size_t i = 0; i < status.size(); ++i) {
                if (status[i] && std::isfinite(candidatePoints[i].x) && std::isfinite(candidatePoints[i].y)) {
                    trackedOld.push_back(previousPoints_[i]);
                    trackedNew.push_back(candidatePoints[i]);
                    trackedIds.push_back(previousLandmarkIds_[i]);
                }
            }

            if (trackedNew.size() >= static_cast<std::size_t>(config_.minInliers)) {
                // Geometric motion estimation
                cv::Mat mask;
                const cv::Mat K = cv::Mat(camera_.matrix());
                const cv::Mat essential = cv::findEssentialMat(trackedOld, trackedNew, K, cv::RANSAC, 0.999, 1.0, mask);
                if (!essential.empty()) {
                    cv::Mat R, t;
                    cv::recoverPose(essential, trackedOld, trackedNew, K, R, t, mask);
                    for (int i = 0; i < mask.rows * mask.cols; ++i) {
                        if (mask.ptr<unsigned char>()[i] != 0) ++visualInliers;
                    }
                    frameMacs += trackedOld.size() * 120; // Essential matrix RANSAC MACs
                }
            }

            // Extract binary descriptors and update hardware HBST tree (0 float MACs)
            if (config_.enableHbstMatching && !trackedNew.empty()) {
                const auto descriptors = extractBinaryDescriptors(processedImage, trackedNew);
                if (landmarkDescriptors_.empty()) {
                    landmarkDescriptors_ = descriptors;
                    landmarkIndex_.build(landmarkDescriptors_);
                }
            }

            previousRawImage_ = processedImage.clone();
            previousPoints_ = trackedNew;
            previousLandmarkIds_ = trackedIds;

            // Calculate visual confidence
            const double trackedRatio = previousPoints_.empty() ? 0.0 :
                static_cast<double>(visualInliers) / static_cast<double>(previousPoints_.size());
            state_.confidence = std::clamp(0.40 * trackedRatio + 0.60 * (visualInliers >= 12 ? 1.0 : 0.0), 0.0, 1.0);
            state_.visualInliers = visualInliers;
            state_.lidarInliers = lidarInliers;
        }
    }

    const auto endTime = std::chrono::steady_clock::now();
    state_.elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Power & Energy estimation:
    // Dynamic MAC energy on 16nm SoC: ~0.5 pJ / MAC. Static base power: ~1.2 W.
    state_.estimatedMacCount = frameMacs;
    totalMacs_ += frameMacs;
    const double frameEnergyJoules = (frameMacs * 0.5e-12) + (1.2 * state_.elapsedMs * 1e-3);
    state_.estimatedEnergyJoules = frameEnergyJoules;
    totalJoules_ += frameEnergyJoules;

    trajectory_.push_back(state_);
    return state_;
}

cv::Mat applyIlluminationShock(const cv::Mat& input, const std::string& shockType) {
    if (shockType == "blackout" || shockType == "tunnel_darkness") {
        // Complete zero-lux pitch darkness
        return cv::Mat::zeros(input.size(), input.type());
    }
    if (shockType == "solar_flare" || shockType == "glare") {
        // Intense direct sun saturation
        return cv::Mat(input.size(), input.type(), cv::Scalar(255));
    }
    if (shockType == "strobe") {
        // Periodic high-frequency flicker
        cv::Mat output;
        input.convertTo(output, input.type(), 0.1, 10.0);
        return output;
    }
    return input.clone();
}

} // namespace adaptive_slam
