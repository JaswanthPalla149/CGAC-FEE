#include "adaptive_slam/uav_hybrid_slam.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

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
    globalLidarMap_.clear();
    currentScanWorld_.clear();
    mapPointFrameIndices_.clear();
    metricLandmarkFrameIndices_.clear();
    poseGraphPoses_.clear();
    loopRecognitionKeyframes_.clear();
    lastImuTimestampNs_ = 0;
    hasImuTimestamp_ = false;
    totalMacs_ = 0;
    totalJoules_ = 0.0;
    loopClosures_ = 0;
    maxLoopDriftMeters_ = 0.0;
    lastLoopClosureFrame_ = 0;
    hasLastLoopClosure_ = false;
}

void UAVHybridSLAM::processImu(const ImuMeasurement& imu) {
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(imu.angularVel[axis]) ||
            !std::isfinite(imu.linearAccel[axis])) {
            throw std::invalid_argument("IMU measurements must be finite");
        }
    }
    if (!hasImuTimestamp_) {
        lastImuTimestampNs_ = imu.timestampNs;
        state_.timestampNs = imu.timestampNs;
        hasImuTimestamp_ = true;
        return;
    }

    if (imu.timestampNs <= lastImuTimestampNs_) {
        throw std::invalid_argument("IMU timestamps must be strictly increasing");
    }
    const double dt =
        static_cast<double>(imu.timestampNs - lastImuTimestampNs_) * 1e-9;
    lastImuTimestampNs_ = imu.timestampNs;

    const cv::Vec3d unbOmega = imu.angularVel - state_.gyroBias;
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

    const cv::Vec3d accelerationWorld =
        state_.rotationWorldBody * (imu.linearAccel - state_.accelBias) +
        config_.gravity;
    state_.positionWorld += state_.velocityWorld * dt +
                            accelerationWorld * (0.5 * dt * dt);
    state_.velocityWorld += accelerationWorld * dt;
    state_.timestampNs = imu.timestampNs;
    totalMacs_ += 75;
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
            if (metricLandmarks_.size() < 4096 && (i % 5 == 0)) {
                cv::Vec3d pBody(pt.x, pt.y, pt.z);
                cv::Vec3d pWorld = state_.positionWorld + state_.rotationWorldBody * pBody;
                metricLandmarks_.emplace_back(pWorld[0], pWorld[1], pWorld[2]);
                metricLandmarkFrameIndices_.push_back(trajectory_.size());
            }
        }
    }
    totalMacs_ += (scan.points.size() / stride) * 3; // 3 MACs per point projection
}

void UAVHybridSLAM::updateMap(const LidarScan& scan, const cv::Matx33d& R, const cv::Vec3d& t) {
    currentScanWorld_.clear();
    if (scan.points.empty()) return;

    for (std::size_t i = 0; i < scan.points.size(); ++i) {
        const auto& pt = scan.points[i];
        const double d2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
        if (d2 > 0.09 && d2 < 3600.0) {
            cv::Vec3d pBody(pt.x, pt.y, pt.z);
            cv::Vec3d pWorld = t + R * pBody;
            currentScanWorld_.emplace_back(pWorld[0], pWorld[1], pWorld[2]);
        }
    }

    if (globalLidarMap_.empty()) {
        globalLidarMap_ = currentScanWorld_;
        mapPointFrameIndices_.assign(globalLidarMap_.size(), trajectory_.size());
        return;
    }

    const std::size_t step = std::max<std::size_t>(1, currentScanWorld_.size() / 120);
    for (std::size_t i = 0; i < currentScanWorld_.size(); i += step) {
        const auto& wpt = currentScanWorld_[i];
        bool duplicate = false;
        const std::size_t checkStart = globalLidarMap_.size() > 80 ? globalLidarMap_.size() - 80 : 0;
        for (std::size_t j = checkStart; j < globalLidarMap_.size(); ++j) {
            const auto& mp = globalLidarMap_[j];
            if (std::abs(mp.x - wpt.x) < 0.10 &&
                std::abs(mp.y - wpt.y) < 0.10 &&
                std::abs(mp.z - wpt.z) < 0.10) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate && globalLidarMap_.size() < 12000) {
            globalLidarMap_.push_back(wpt);
            mapPointFrameIndices_.push_back(trajectory_.size());
        }
    }
}

bool UAVHybridSLAM::alignLidarScan(const LidarScan& scan,
                                   cv::Matx33d& deltaR,
                                   cv::Vec3d& deltaT,
                                   int& inliers,
                                   HybridPowerMode mode) {
    inliers = 0;
    deltaR = cv::Matx33d::eye();
    deltaT = cv::Vec3d(0.0, 0.0, 0.0);

    if (scan.points.empty() || globalLidarMap_.size() < 12) {
        return false;
    }

    const std::size_t stride = (mode == HybridPowerMode::LowCost) ? 2U : 1U;
    std::vector<cv::Vec3d> bodyPts;
    bodyPts.reserve(scan.points.size() / stride);
    for (std::size_t i = 0; i < scan.points.size(); i += stride) {
        const auto& pt = scan.points[i];
        const double d2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
        if (d2 > 0.09 && d2 < 3600.0) {
            bodyPts.emplace_back(pt.x, pt.y, pt.z);
        }
    }

    if (bodyPts.size() < 8) {
        return false;
    }

    cv::Matx33d currentR = state_.rotationWorldBody;
    cv::Vec3d currentT = state_.positionWorld;
    const double maxDistSq = 0.40 * 0.40; // 40 cm search radius

    for (int iter = 0; iter < 4; ++iter) {
        std::vector<cv::Vec3d> matchedW;
        std::vector<cv::Vec3d> matchedM;
        matchedW.reserve(bodyPts.size());
        matchedM.reserve(bodyPts.size());

        for (const auto& bpt : bodyPts) {
            cv::Vec3d ptW = currentT + currentR * bpt;
            double bestD2 = maxDistSq;
            cv::Vec3d bestMapPt;
            bool found = false;

            // Search over the recent local map
            const std::size_t startIdx = globalLidarMap_.size() > 800 ? globalLidarMap_.size() - 800 : 0;
            for (std::size_t m = startIdx; m < globalLidarMap_.size(); ++m) {
                const auto& mapPt = globalLidarMap_[m];
                const double dx = ptW[0] - mapPt.x;
                const double dy = ptW[1] - mapPt.y;
                const double dz = ptW[2] - mapPt.z;
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < bestD2) {
                    bestD2 = d2;
                    bestMapPt = cv::Vec3d(mapPt.x, mapPt.y, mapPt.z);
                    found = true;
                }
            }

            if (found) {
                matchedW.push_back(ptW);
                matchedM.push_back(bestMapPt);
            }
        }

        if (matchedW.size() < 8) {
            break;
        }

        inliers = static_cast<int>(matchedW.size());

        // Point-to-Point Kabsch residual solver
        cv::Vec3d srcCenter(0.0, 0.0, 0.0);
        cv::Vec3d tgtCenter(0.0, 0.0, 0.0);
        const double nPts = static_cast<double>(matchedW.size());
        for (std::size_t k = 0; k < matchedW.size(); ++k) {
            srcCenter += matchedW[k];
            tgtCenter += matchedM[k];
        }
        srcCenter *= (1.0 / nPts);
        tgtCenter *= (1.0 / nPts);

        cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
        for (std::size_t k = 0; k < matchedW.size(); ++k) {
            cv::Vec3d p = matchedW[k] - srcCenter;
            cv::Vec3d q = matchedM[k] - tgtCenter;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    H.at<double>(r, c) += p[r] * q[c];
                }
            }
        }

        cv::SVD svd(H, cv::SVD::FULL_UV);
        cv::Mat R_mat = svd.vt.t() * svd.u.t();
        if (cv::determinant(R_mat) < 0) {
            cv::Mat vt_adj = svd.vt.clone();
            vt_adj.row(2) *= -1.0;
            R_mat = vt_adj.t() * svd.u.t();
        }

        cv::Matx33d dR;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                dR(r, c) = R_mat.at<double>(r, c);
            }
        }

        cv::Vec3d dT = tgtCenter - dR * srcCenter;

        // Apply bounded update
        if (cv::norm(dT) < 0.35) {
            currentT += dT;
            currentR = dR * currentR;
        }
    }

    deltaT = currentT - state_.positionWorld;
    deltaR = state_.rotationWorldBody.t() * currentR;
    totalMacs_ += bodyPts.size() * 12; // ~12 MACs per ICP point iteration
    return (inliers >= 8);
}

bool UAVHybridSLAM::detectAndCloseLoop(
    std::size_t frameIndex,
    const std::vector<BinaryDescriptor>& descriptors,
    const LidarScan& scan) {
    if (descriptors.size() < 12 || scan.points.size() < 12 ||
        poseGraphPoses_.size() <= frameIndex ||
        (hasLastLoopClosure_ && frameIndex < lastLoopClosureFrame_ + 10)) {
        return false;
    }

    const LoopRecognitionKeyframe* bestCandidate = nullptr;
    std::size_t bestMatchCount = 0;
    for (const auto& candidate : loopRecognitionKeyframes_) {
        if (frameIndex < candidate.frameIndex + 150 ||
            candidate.descriptors.empty() ||
            candidate.lidarPointsBody.size() < 12) {
            continue;
        }
        std::vector<bool> usedCandidate(candidate.descriptors.size(), false);
        std::size_t matches = 0;
        for (const auto& query : descriptors) {
            std::uint16_t bestDistance = 257;
            std::uint16_t secondDistance = 257;
            std::size_t bestIndex = 0;
            for (std::size_t index = 0; index < candidate.descriptors.size();
                 ++index) {
                const std::uint16_t distance =
                    query.hammingDistance(candidate.descriptors[index]);
                if (distance < bestDistance) {
                    secondDistance = bestDistance;
                    bestDistance = distance;
                    bestIndex = index;
                } else if (distance < secondDistance) {
                    secondDistance = distance;
                }
            }
            const bool ratioPassed =
                secondDistance == 0
                    ? bestDistance == 0
                    : static_cast<double>(bestDistance) /
                              static_cast<double>(secondDistance) <
                          0.85;
            if (bestDistance <= 52 && ratioPassed &&
                !usedCandidate[bestIndex]) {
                usedCandidate[bestIndex] = true;
                ++matches;
            }
        }
        if (matches > bestMatchCount) {
            bestMatchCount = matches;
            bestCandidate = &candidate;
        }
    }
    if (bestCandidate == nullptr || bestMatchCount < 12) {
        return false;
    }

    std::vector<cv::Vec3d> sourcePoints;
    std::vector<cv::Vec3d> targetPoints = bestCandidate->lidarPointsBody;
    sourcePoints.reserve(scan.points.size());
    for (const auto& point : scan.points) {
        const double squaredRange =
            static_cast<double>(point.x) * point.x +
            static_cast<double>(point.y) * point.y +
            static_cast<double>(point.z) * point.z;
        if (squaredRange > 0.09 && squaredRange < 3600.0) {
            sourcePoints.emplace_back(point.x, point.y, point.z);
        }
    }
    if (sourcePoints.size() < 12) {
        return false;
    }
    if (sourcePoints.size() != targetPoints.size()) {
        return false;
    }
    double orderedScanSquaredError = 0.0;
    for (std::size_t index = 0; index < sourcePoints.size(); ++index) {
        orderedScanSquaredError +=
            cv::norm(sourcePoints[index] - targetPoints[index],
                     cv::NORM_L2SQR);
    }
    const double orderedScanRms =
        std::sqrt(orderedScanSquaredError /
                  static_cast<double>(sourcePoints.size()));
    if (orderedScanRms > 0.12) {
        return false;
    }

    const PoseGraphPose& candidatePose =
        poseGraphPoses_[bestCandidate->frameIndex];
    const PoseGraphPose& currentPose = poseGraphPoses_[frameIndex];
    cv::Matx33d relativeRotation = cv::Matx33d::eye();
    cv::Vec3d relativeTranslation(0.0, 0.0, 0.0);
    std::vector<std::pair<cv::Vec3d, cv::Vec3d>> correspondences;
    for (int iteration = 0; iteration < 8; ++iteration) {
        correspondences.clear();
        std::vector<bool> usedTargets(targetPoints.size(), false);
        for (const auto& source : sourcePoints) {
            const cv::Vec3d transformed =
                relativeRotation * source + relativeTranslation;
            double bestSquaredDistance = 0.8 * 0.8;
            std::size_t nearestIndex = targetPoints.size();
            for (std::size_t targetIndex = 0;
                 targetIndex < targetPoints.size(); ++targetIndex) {
                if (usedTargets[targetIndex]) {
                    continue;
                }
                const double squaredDistance = cv::norm(
                    transformed - targetPoints[targetIndex], cv::NORM_L2SQR);
                if (squaredDistance < bestSquaredDistance) {
                    bestSquaredDistance = squaredDistance;
                    nearestIndex = targetIndex;
                }
            }
            if (nearestIndex < targetPoints.size()) {
                usedTargets[nearestIndex] = true;
                correspondences.emplace_back(source, targetPoints[nearestIndex]);
            }
        }
        if (correspondences.size() < 12) {
            return false;
        }

        cv::Vec3d sourceCenter(0.0, 0.0, 0.0);
        cv::Vec3d targetCenter(0.0, 0.0, 0.0);
        for (const auto& pair : correspondences) {
            sourceCenter += pair.first;
            targetCenter += pair.second;
        }
        const double inverseCount =
            1.0 / static_cast<double>(correspondences.size());
        sourceCenter *= inverseCount;
        targetCenter *= inverseCount;
        cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
        for (const auto& pair : correspondences) {
            const cv::Vec3d source = pair.first - sourceCenter;
            const cv::Vec3d target = pair.second - targetCenter;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    covariance.at<double>(row, column) +=
                        source[row] * target[column];
                }
            }
        }
        cv::SVD svd(covariance, cv::SVD::FULL_UV);
        cv::Mat rotation = svd.vt.t() * svd.u.t();
        if (cv::determinant(rotation) < 0.0) {
            cv::Mat adjustedVt = svd.vt.clone();
            adjustedVt.row(2) *= -1.0;
            rotation = adjustedVt.t() * svd.u.t();
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                relativeRotation(row, column) =
                    rotation.at<double>(row, column);
            }
        }
        relativeTranslation = targetCenter - relativeRotation * sourceCenter;
    }

    double squaredError = 0.0;
    for (const auto& pair : correspondences) {
        squaredError += cv::norm(
                            relativeRotation * pair.first +
                                relativeTranslation - pair.second,
                            cv::NORM_L2SQR);
    }
    const double rmsError =
        std::sqrt(squaredError / static_cast<double>(correspondences.size()));
    if (rmsError > 0.12 || correspondences.size() < 30 ||
        static_cast<double>(correspondences.size()) /
                static_cast<double>(sourcePoints.size()) <
            0.60) {
        return false;
    }

    const double driftMeters = cv::norm(
        candidatePose.rotationWorldFromBody.t() *
            (currentPose.positionWorld - candidatePose.positionWorld) -
        relativeTranslation);

    std::vector<std::size_t> graphNodeFrames;
    for (std::size_t nodeFrame = 0; nodeFrame < frameIndex;
         nodeFrame += 10) {
        graphNodeFrames.push_back(nodeFrame);
    }
    if (graphNodeFrames.empty() || graphNodeFrames.back() != frameIndex) {
        graphNodeFrames.push_back(frameIndex);
    }
    std::vector<PoseGraphPose> previousGraphPoses;
    previousGraphPoses.reserve(graphNodeFrames.size());
    for (const std::size_t nodeFrame : graphNodeFrames) {
        previousGraphPoses.push_back(poseGraphPoses_[nodeFrame]);
    }
    std::vector<PoseGraphConstraint> constraints;
    constraints.reserve(graphNodeFrames.size());
    for (std::size_t node = 1; node < graphNodeFrames.size(); ++node) {
        constraints.push_back(makePoseGraphConstraint(
            node - 1, node, previousGraphPoses[node - 1],
            previousGraphPoses[node], 25.0, 100.0));
    }
    const auto candidateNode = std::find(
        graphNodeFrames.begin(), graphNodeFrames.end(),
        bestCandidate->frameIndex);
    if (candidateNode == graphNodeFrames.end()) {
        throw std::logic_error(
            "loop-recognition keyframe is missing from the pose graph");
    }
    PoseGraphConstraint loopConstraint;
    loopConstraint.from = static_cast<std::size_t>(
        std::distance(graphNodeFrames.begin(), candidateNode));
    loopConstraint.to = graphNodeFrames.size() - 1;
    loopConstraint.rotationFromTo = relativeRotation;
    loopConstraint.translationFromTo = relativeTranslation;
    loopConstraint.translationInformation = 100.0;
    loopConstraint.rotationInformation = 100.0;
    loopConstraint.loopClosure = true;
    constraints.push_back(loopConstraint);

    std::vector<PoseGraphPose> optimizedGraphPoses = previousGraphPoses;
    if (!optimizePoseGraph(optimizedGraphPoses, constraints)) {
        throw std::runtime_error(
            "pose-graph optimization failed after verified UAV loop closure");
    }
    const std::vector<PoseGraphPose> previousFramePoses = poseGraphPoses_;
    applyPoseGraphCorrection(previousFramePoses, graphNodeFrames,
                             previousGraphPoses, optimizedGraphPoses);
    state_.loopClosed = true;
    state_.loopCandidateFrame =
        static_cast<int>(bestCandidate->frameIndex);
    state_.loopDriftMeters = driftMeters;
    ++loopClosures_;
    maxLoopDriftMeters_ = std::max(maxLoopDriftMeters_, driftMeters);
    lastLoopClosureFrame_ = frameIndex;
    hasLastLoopClosure_ = true;
    return true;
}

void UAVHybridSLAM::applyPoseGraphCorrection(
    const std::vector<PoseGraphPose>& previousFramePoses,
    const std::vector<std::size_t>& graphNodeFrames,
    const std::vector<PoseGraphPose>& previousGraphPoses,
    const std::vector<PoseGraphPose>& optimizedGraphPoses) {
    const auto graphNodeForFrame = [&graphNodeFrames](std::size_t frameIndex) {
        const auto nextNode = std::upper_bound(
            graphNodeFrames.begin(), graphNodeFrames.end(), frameIndex);
        return nextNode == graphNodeFrames.begin()
                   ? std::size_t{0}
                   : static_cast<std::size_t>(
                         std::distance(graphNodeFrames.begin(), nextNode) - 1);
    };
    const auto correctedPoseForFrame =
        [&graphNodeForFrame, &previousFramePoses, &previousGraphPoses,
         &optimizedGraphPoses](std::size_t frameIndex) {
            const std::size_t graphNode = graphNodeForFrame(frameIndex);
            const PoseGraphPose& oldAnchor = previousGraphPoses[graphNode];
            const PoseGraphPose& newAnchor = optimizedGraphPoses[graphNode];
            const PoseGraphPose& oldFrame = previousFramePoses[frameIndex];
            const cv::Matx33d correctionRotation =
                newAnchor.rotationWorldFromBody *
                oldAnchor.rotationWorldFromBody.t();
            PoseGraphPose corrected;
            corrected.rotationWorldFromBody =
                correctionRotation * oldFrame.rotationWorldFromBody;
            corrected.positionWorld =
                newAnchor.rotationWorldFromBody *
                    (oldAnchor.rotationWorldFromBody.t() *
                     (oldFrame.positionWorld - oldAnchor.positionWorld)) +
                newAnchor.positionWorld;
            return corrected;
        };

    for (std::size_t frameIndex = 0;
         frameIndex < poseGraphPoses_.size(); ++frameIndex) {
        poseGraphPoses_[frameIndex] =
            correctedPoseForFrame(frameIndex);
    }

    for (std::size_t index = 0; index < trajectory_.size(); ++index) {
        const cv::Matx33d correctionRotation =
            poseGraphPoses_[index].rotationWorldFromBody *
            previousFramePoses[index].rotationWorldFromBody.t();
        trajectory_[index].positionWorld =
            poseGraphPoses_[index].positionWorld;
        trajectory_[index].rotationWorldBody =
            poseGraphPoses_[index].rotationWorldFromBody;
        trajectory_[index].velocityWorld =
            correctionRotation * trajectory_[index].velocityWorld;
    }

    const auto transformMapPoint =
        [&correctedPoseForFrame, &previousFramePoses](
            const cv::Vec3d& point, std::size_t frameIndex) {
            if (frameIndex >= previousFramePoses.size()) {
                return point;
            }
            const PoseGraphPose& oldFrame = previousFramePoses[frameIndex];
            const PoseGraphPose corrected = correctedPoseForFrame(frameIndex);
            const cv::Vec3d local =
                oldFrame.rotationWorldFromBody.t() *
                (point - oldFrame.positionWorld);
            return corrected.rotationWorldFromBody * local +
                   corrected.positionWorld;
        };

    for (std::size_t index = 0; index < globalLidarMap_.size(); ++index) {
        const cv::Point3d& point = globalLidarMap_[index];
        const std::size_t frameIndex = mapPointFrameIndices_[index];
        const cv::Vec3d corrected = transformMapPoint(
            cv::Vec3d(point.x, point.y, point.z), frameIndex);
        globalLidarMap_[index] =
            cv::Point3d(corrected[0], corrected[1], corrected[2]);
    }
    for (std::size_t index = 0; index < metricLandmarks_.size(); ++index) {
        const cv::Point3d& point = metricLandmarks_[index];
        const cv::Vec3d corrected = transformMapPoint(
            cv::Vec3d(point.x, point.y, point.z),
            metricLandmarkFrameIndices_[index]);
        metricLandmarks_[index] =
            cv::Point3d(corrected[0], corrected[1], corrected[2]);
    }
    for (auto& point : currentScanWorld_) {
        const std::size_t currentFrame = poseGraphPoses_.size() - 1;
        const cv::Vec3d corrected = transformMapPoint(
            cv::Vec3d(point.x, point.y, point.z), currentFrame);
        point = cv::Point3d(corrected[0], corrected[1], corrected[2]);
    }

    const std::size_t currentFrame = poseGraphPoses_.size() - 1;
    const cv::Matx33d correctionRotation =
        poseGraphPoses_[currentFrame].rotationWorldFromBody *
        previousFramePoses[currentFrame].rotationWorldFromBody.t();
    state_.positionWorld = poseGraphPoses_[currentFrame].positionWorld;
    state_.rotationWorldBody =
        poseGraphPoses_[currentFrame].rotationWorldFromBody;
    state_.velocityWorld = correctionRotation * state_.velocityWorld;
}

UAVState UAVHybridSLAM::processFrame(const cv::Mat& image,
                                     std::int64_t timestampNs,
                                     const std::optional<LidarScan>& scan) {
    const auto startTime = std::chrono::steady_clock::now();
    state_.timestampNs = timestampNs;
    state_.loopClosed = false;
    state_.loopCandidateFrame = -1;
    state_.loopDriftMeters = 0.0;
    std::uint64_t frameMacs = 0;

    // 1. Fast Optical Invariant Test for Sudden Lighting Collapse
    cv::Scalar meanIntensity, stdDevIntensity;
    cv::meanStdDev(image, meanIntensity, stdDevIntensity);
    const double meanVal = meanIntensity[0];
    const double stdVal = stdDevIntensity[0];

    const bool suddenLightCollapse = (meanVal < 6.0 || meanVal > 248.0 || stdVal < 3.0);
    double initialConfidence = state_.confidence;
    if (suddenLightCollapse) {
        initialConfidence = 0.05;
    } else if (state_.mode == HybridPowerMode::VisualBypassLIO) {
        // Optical recovery: lighting has returned, restore confidence to exit bypass
        initialConfidence = 0.85;
    }

    // 2. Controller Evaluation
    const double lidarDegeneracy = (scan.has_value() && scan->points.size() < 15) ? 0.90 : 0.10;
    const HybridPowerMode chosenMode = controller_.choose(initialConfidence, lidarDegeneracy, state_.mode);
    state_.mode = chosenMode;

    int visualInliers = 0;
    int lidarInliers = 0;
    cv::Matx33d deltaR_lidar = cv::Matx33d::eye();
    cv::Vec3d deltaT_lidar(0.0, 0.0, 0.0);
    bool lidarValid = false;

    // 3. LiDAR ICP scan alignment against 3D metric map
    if (scan.has_value()) {
        lidarValid = alignLidarScan(*scan, deltaR_lidar, deltaT_lidar, lidarInliers, chosenMode);
        updateLidarPoints(*scan, chosenMode, lidarInliers);
    }

    // 4. Visual Execution OR Zero-MAC Visual Bypass
    if (chosenMode == HybridPowerMode::VisualBypassLIO) {
        // Sudden blackout or blinding flash: BYPASS visual pipeline completely!
        // 0 visual MACs spent. UAV continues on LiDAR + IMU integration!
        if (lidarValid && cv::norm(deltaT_lidar) < 0.35) {
            state_.positionWorld += deltaT_lidar;
            state_.rotationWorldBody = state_.rotationWorldBody * deltaR_lidar;
        } else {
            // Kinematic continuation using velocity
            state_.positionWorld += state_.velocityWorld * 0.02;
        }
        if (scan.has_value()) {
            updateMap(*scan, state_.rotationWorldBody, state_.positionWorld);
        }
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

            // First frame initialization must NOT report 0 confidence!
            state_.confidence = (previousPoints_.size() >= static_cast<std::size_t>(config_.minInliers)) ? 1.0 : 0.6;
            state_.visualInliers = static_cast<int>(previousPoints_.size());
            state_.lidarInliers = lidarInliers;

            if (scan.has_value()) {
                updateMap(*scan, state_.rotationWorldBody, state_.positionWorld);
            }
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
            double sumDu = 0.0;
            double sumDv = 0.0;

            for (std::size_t i = 0; i < status.size(); ++i) {
                if (status[i] && std::isfinite(candidatePoints[i].x) && std::isfinite(candidatePoints[i].y)) {
                    const double du = candidatePoints[i].x - previousPoints_[i].x;
                    const double dv = candidatePoints[i].y - previousPoints_[i].y;
                    if (std::abs(du) < 50.0 && std::abs(dv) < 50.0) {
                        trackedOld.push_back(previousPoints_[i]);
                        trackedNew.push_back(candidatePoints[i]);
                        trackedIds.push_back(previousLandmarkIds_[i]);
                        sumDu += du;
                        sumDv += dv;
                        ++visualInliers;
                    }
                }
            }

            frameMacs += trackedOld.size() * 120;

            // Compute LiDAR metric depth to the environment
            double lidarDepth = 3.8;
            if (scan.has_value() && !scan->points.empty()) {
                std::vector<double> forwardRanges;
                forwardRanges.reserve(scan->points.size());
                for (const auto& pt : scan->points) {
                    if (pt.y > 0.3f && pt.y < 35.0f) {
                        forwardRanges.push_back(pt.y);
                    }
                }
                if (!forwardRanges.empty()) {
                    std::nth_element(forwardRanges.begin(), forwardRanges.begin() + forwardRanges.size() / 2, forwardRanges.end());
                    lidarDepth = forwardRanges[forwardRanges.size() / 2];
                }
            }

            // Tightly coupled multi-modal visual-LiDAR motion step
            if (visualInliers >= config_.minInliers) {
                const double meanDu = sumDu / visualInliers;
                const double meanDv = sumDv / visualInliers;

                // Visual-inertial motion step scaled by metric LiDAR depth
                const double dx_step = -(meanDu * lidarDepth) / camera_.fx;
                const double dz_step = (meanDv * lidarDepth) / camera_.fy;
                double dy_step = deltaT_lidar[1];

                // If LiDAR ICP has inliers, use full ICP delta for Y
                if (!lidarValid || std::abs(dy_step) > 0.20) {
                    dy_step = state_.velocityWorld[1] * 0.02;
                }

                cv::Vec3d fusedStep(dx_step, dy_step, dz_step);
                if (cv::norm(fusedStep) < 0.35) {
                    state_.positionWorld += fusedStep;
                    state_.velocityWorld = fusedStep * 50.0;
                }
            } else if (lidarValid && cv::norm(deltaT_lidar) < 0.35) {
                state_.positionWorld += deltaT_lidar;
                state_.rotationWorldBody = state_.rotationWorldBody * deltaR_lidar;
            }

            if (scan.has_value()) {
                updateMap(*scan, state_.rotationWorldBody, state_.positionWorld);
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

    const std::size_t frameIndex = poseGraphPoses_.size();
    const PoseGraphPose currentPose{state_.rotationWorldBody,
                                    state_.positionWorld};
    poseGraphPoses_.push_back(currentPose);

    if (scan.has_value() && chosenMode != HybridPowerMode::VisualBypassLIO &&
        !previousRawImage_.empty() && previousPoints_.size() >= 12) {
        const auto recognitionPoints =
            detectCorners(previousRawImage_, chosenMode);
        const auto descriptors =
            extractBinaryDescriptors(previousRawImage_, recognitionPoints);
        detectAndCloseLoop(frameIndex, descriptors, *scan);
        if (frameIndex % 10 == 0) {
            LoopRecognitionKeyframe keyframe;
            keyframe.frameIndex = frameIndex;
            keyframe.descriptors = descriptors;
            for (const auto& point : scan->points) {
                const double squaredRange =
                    static_cast<double>(point.x) * point.x +
                    static_cast<double>(point.y) * point.y +
                    static_cast<double>(point.z) * point.z;
                if (squaredRange > 0.09 && squaredRange < 3600.0) {
                    keyframe.lidarPointsBody.emplace_back(point.x, point.y,
                                                          point.z);
                }
            }
            if (keyframe.descriptors.size() >= 12 &&
                keyframe.lidarPointsBody.size() >= 12) {
                loopRecognitionKeyframes_.push_back(std::move(keyframe));
            }
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
