#include "adaptive_slam/slam.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace adaptive_slam {
namespace {

struct FeatureSettings {
    int maxCorners;
    double quality;
    cv::Size window;
    int pyramidLevels;
};

FeatureSettings settingsFor(TrackingMode mode) {
    if (mode == TrackingMode::Recovery) {
        return {1400, 0.008, cv::Size(31, 31), 4};
    }
    return {450, 0.015, cv::Size(21, 21), 2};
}

cv::Mat cameraMatrix(const CameraModel& camera) {
    return cv::Mat(camera.matrix()).clone();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 != 0) {
        return *middle;
    }
    const double upper = *middle;
    const auto lower = std::max_element(values.begin(), middle);
    return (*lower + upper) * 0.5;
}

cv::Mat projectionMatrix(const CameraModel& camera,
                         const cv::Matx33d& rotationCameraWorld,
                         const cv::Vec3d& translationCameraWorld) {
    cv::Matx34d extrinsic;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            extrinsic(row, column) = rotationCameraWorld(row, column);
        }
        extrinsic(row, 3) = translationCameraWorld[row];
    }
    return cv::Mat(camera.matrix() * extrinsic).clone();
}

}  // namespace

const char* modeName(TrackingMode mode) noexcept {
    return mode == TrackingMode::Recovery ? "recovery" : "low_cost";
}

cv::Matx33d CameraModel::matrix() const {
    return cv::Matx33d(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
}

FeatureSLAM::FeatureSLAM(CameraModel camera, int minInliers,
                         int keyframeInterval)
    : camera_(camera),
      minInliers_(minInliers),
      keyframeInterval_(keyframeInterval) {
    if (!(camera_.fx > 0.0 && camera_.fy > 0.0) || minInliers_ < 5 ||
        keyframeInterval_ < 1) {
        throw std::invalid_argument("invalid camera or SLAM configuration");
    }
    landmarks_.reserve(4096);
    keyframes_.reserve(256);
    previousPoints_.reserve(1400);
    previousLandmarkIds_.reserve(1400);
}

cv::Mat FeatureSLAM::prepare(const cv::Mat& image, TrackingMode mode) const {
    if (image.empty()) {
        throw std::invalid_argument("cannot process an empty image");
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("image must have 1, 3, or 4 channels");
    }
    if (mode == TrackingMode::LowCost) {
        return gray;
    }
    cv::Mat enhanced;
    static thread_local cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, enhanced);
    return enhanced;
}

std::vector<cv::Point2f> FeatureSLAM::detect(const cv::Mat& image,
                                             TrackingMode mode) const {
    const FeatureSettings settings = settingsFor(mode);
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(image, points, settings.maxCorners, settings.quality,
                            7.0, cv::noArray(), 7, false, 0.04);
    return points;
}

FrameResult FeatureSLAM::process(const cv::Mat& image,
                                 std::int64_t timestampNs,
                                 TrackingMode mode) {
    const auto start = std::chrono::steady_clock::now();
    cv::Mat rawGray;
    if (image.channels() == 1) {
        rawGray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, rawGray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, rawGray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("image must have 1, 3, or 4 channels");
    }
    const cv::Mat currentImage = prepare(rawGray, mode);
    FrameResult result;
    result.timestampNs = timestampNs;
    result.mode = mode;

    if (previousRawImage_.empty()) {
        previousRawImage_ = rawGray.clone();
        previousPoints_ = detect(currentImage, mode);
        previousLandmarkIds_.assign(previousPoints_.size(), -1);
        keyframes_.push_back({timestampNs, rotationWorldCamera_, positionWorld_});
        result.position = positionWorld_;
        result.confidence = 1.0;
        result.trackedRatio = 1.0;
        result.activeFeatures = previousPoints_.size();
        result.elapsedMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      start)
                .count();
        return result;
    }

    const cv::Mat previousImage = prepare(previousRawImage_, mode);
    if (previousPoints_.size() < 5) {
        previousPoints_ = detect(previousImage, mode);
        previousLandmarkIds_.assign(previousPoints_.size(), -1);
    }
    if (previousLandmarkIds_.size() != previousPoints_.size()) {
        throw std::logic_error("feature and landmark association state is inconsistent");
    }

    const cv::Matx33d previousRotation = rotationWorldCamera_;
    const cv::Vec3d previousPosition = positionWorld_;
    std::vector<cv::Point2f> oldTracked;
    std::vector<cv::Point2f> newTracked;
    std::vector<int> trackedIds;
    std::vector<int> trackedStatus;
    double inlierRatio = 0.0;
    double reprojectionError = std::numeric_limits<double>::infinity();
    int inlierCount = 0;
    int mapInlierCount = 0;
    bool poseUpdated = false;

    if (previousPoints_.size() >= 5) {
        std::vector<cv::Point2f> candidatePoints;
        std::vector<unsigned char> status;
        std::vector<float> errors;
        const FeatureSettings settings = settingsFor(mode);
        cv::calcOpticalFlowPyrLK(
            previousImage, currentImage, previousPoints_, candidatePoints, status,
            errors, settings.window, settings.pyramidLevels,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30,
                             0.01));
        oldTracked.reserve(status.size());
        newTracked.reserve(status.size());
        trackedIds.reserve(status.size());
        for (std::size_t i = 0; i < status.size(); ++i) {
            if (status[i] != 0 && std::isfinite(candidatePoints[i].x) &&
                std::isfinite(candidatePoints[i].y)) {
                oldTracked.push_back(previousPoints_[i]);
                newTracked.push_back(candidatePoints[i]);
                trackedIds.push_back(previousLandmarkIds_[i]);
            }
        }
        result.trackedRatio =
            static_cast<double>(newTracked.size()) /
            static_cast<double>(std::max<std::size_t>(1, previousPoints_.size()));
        if (newTracked.size() >= 5) {
            const auto mapPose = estimatePoseFromMap(newTracked, trackedIds);
            if (mapPose.has_value()) {
                inlierRatio = mapPose->inlierRatio;
                reprojectionError = mapPose->medianError;
                inlierCount = mapPose->inlierCount;
                mapInlierCount = mapPose->inlierCount;
                poseUpdated = true;
            } else {
                poseUpdated = estimateMotion(oldTracked, newTracked, inlierRatio,
                                             reprojectionError, inlierCount);
            }
        }
    }

    if (poseUpdated) {
        triangulateNewLandmarks(oldTracked, newTracked, trackedIds,
                                previousRotation, previousPosition);
    }

    const int maxCorners = settingsFor(mode).maxCorners;
    std::vector<cv::Point2f> nextPoints = newTracked;
    std::vector<int> nextIds = trackedIds;
    if (nextPoints.size() < static_cast<std::size_t>(maxCorners * 0.85)) {
        const std::vector<cv::Point2f> candidates = detect(currentImage, mode);
        const float minDistanceSquared = 49.0F;
        for (const cv::Point2f& candidate : candidates) {
            if (nextPoints.size() >= static_cast<std::size_t>(maxCorners)) {
                break;
            }
            const bool separated = std::all_of(
                nextPoints.begin(), nextPoints.end(),
                [&candidate, minDistanceSquared](const cv::Point2f& accepted) {
                    const cv::Point2f delta = accepted - candidate;
                    return delta.dot(delta) >= minDistanceSquared;
                });
            if (separated) {
                nextPoints.push_back(candidate);
                nextIds.push_back(-1);
            }
        }
    }

    previousRawImage_ = rawGray.clone();
    previousPoints_.swap(nextPoints);
    previousLandmarkIds_.swap(nextIds);
    ++frameIndex_;
    if (frameIndex_ % keyframeInterval_ == 0) {
        keyframes_.push_back({timestampNs, rotationWorldCamera_, positionWorld_});
    }

    result.position = positionWorld_;
    result.confidence = confidenceScore(result.trackedRatio, inlierRatio,
                                        reprojectionError);
    result.inlierRatio = inlierRatio;
    result.reprojectionErrorPx = reprojectionError;
    result.activeFeatures = previousPoints_.size();
    result.inliers = inlierCount;
    result.mapInliers = mapInlierCount;
    result.elapsedMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  start)
            .count();
    return result;
}

bool FeatureSLAM::estimateMotion(
    const std::vector<cv::Point2f>& oldPoints,
    const std::vector<cv::Point2f>& newPoints, double& inlierRatio,
    double& reprojectionError, int& inlierCount) {
    inlierRatio = 0.0;
    reprojectionError = std::numeric_limits<double>::infinity();
    inlierCount = 0;
    if (oldPoints.size() < 5 || oldPoints.size() != newPoints.size()) {
        return false;
    }
    cv::Mat mask;
    const cv::Mat essential = cv::findEssentialMat(
        oldPoints, newPoints, cameraMatrix(camera_), cv::RANSAC, 0.999, 1.0,
        mask);
    if (essential.empty()) {
        return false;
    }
    cv::Mat rotation;
    cv::Mat translation;
    cv::recoverPose(essential, oldPoints, newPoints, cameraMatrix(camera_),
                    rotation, translation, mask);
    mask = mask.reshape(1, 1);
    for (int i = 0; i < mask.rows * mask.cols; ++i) {
        inlierCount += mask.ptr<unsigned char>()[i] != 0 ? 1 : 0;
    }
    inlierRatio =
        static_cast<double>(inlierCount) / static_cast<double>(oldPoints.size());

    cv::Mat essentialBlock = essential.rowRange(0, 3);
    cv::Mat essential64;
    essentialBlock.convertTo(essential64, CV_64F);
    const cv::Matx33d essentialMatrix(
        essential64.at<double>(0, 0), essential64.at<double>(0, 1),
        essential64.at<double>(0, 2), essential64.at<double>(1, 0),
        essential64.at<double>(1, 1), essential64.at<double>(1, 2),
        essential64.at<double>(2, 0), essential64.at<double>(2, 1),
        essential64.at<double>(2, 2));
    std::vector<double> errors;
    errors.reserve(static_cast<std::size_t>(inlierCount));
    for (std::size_t i = 0; i < oldPoints.size(); ++i) {
        if (mask.ptr<unsigned char>()[i] == 0) {
            continue;
        }
        const cv::Vec3d first(
            (oldPoints[i].x - camera_.cx) / camera_.fx,
            (oldPoints[i].y - camera_.cy) / camera_.fy, 1.0);
        const cv::Vec3d second(
            (newPoints[i].x - camera_.cx) / camera_.fx,
            (newPoints[i].y - camera_.cy) / camera_.fy, 1.0);
        const cv::Vec3d lineSecond = essentialMatrix * first;
        const cv::Vec3d lineFirst = essentialMatrix.t() * second;
        const double residual = second.dot(lineSecond);
        const double denominator =
            lineFirst[0] * lineFirst[0] + lineFirst[1] * lineFirst[1] +
            lineSecond[0] * lineSecond[0] + lineSecond[1] * lineSecond[1];
        errors.push_back(0.5 * (camera_.fx + camera_.fy) *
                         std::sqrt(residual * residual /
                                   std::max(denominator, 1e-12)));
    }
    reprojectionError = median(std::move(errors));

    if (inlierCount < minInliers_) {
        return false;
    }
    cv::Mat rotation64;
    cv::Mat translation64;
    rotation.convertTo(rotation64, CV_64F);
    translation.convertTo(translation64, CV_64F);
    cv::Matx33d relativeRotation = cv::Matx33d::eye();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            relativeRotation(row, column) = rotation64.at<double>(row, column);
        }
    }
    const cv::Vec3d relativeTranslation(
        translation64.at<double>(0), translation64.at<double>(1),
        translation64.at<double>(2));
    positionWorld_ += rotationWorldCamera_ *
                      (-relativeRotation.t() * relativeTranslation);
    rotationWorldCamera_ = rotationWorldCamera_ * relativeRotation.t();
    return true;
}

std::optional<FeatureSLAM::PoseEstimate> FeatureSLAM::estimatePoseFromMap(
    const std::vector<cv::Point2f>& imagePoints,
    const std::vector<int>& landmarkIds) {
    std::vector<cv::Point3f> worldPoints;
    std::vector<cv::Point2f> observations;
    worldPoints.reserve(landmarkIds.size());
    observations.reserve(landmarkIds.size());
    for (std::size_t i = 0; i < landmarkIds.size(); ++i) {
        const int id = landmarkIds[i];
        if (id < 0 || static_cast<std::size_t>(id) >= landmarks_.size()) {
            continue;
        }
        const cv::Point3d& point = landmarks_[static_cast<std::size_t>(id)];
        worldPoints.emplace_back(static_cast<float>(point.x),
                                 static_cast<float>(point.y),
                                 static_cast<float>(point.z));
        observations.push_back(imagePoints[i]);
    }
    if (worldPoints.size() < static_cast<std::size_t>(minInliers_)) {
        return std::nullopt;
    }

    cv::Mat rotationVector;
    cv::Mat translation;
    std::vector<int> inlierIndices;
    const bool success = cv::solvePnPRansac(
        worldPoints, observations, cameraMatrix(camera_), cv::noArray(),
        rotationVector, translation, false, 100, 3.0F, 0.99, inlierIndices,
        cv::SOLVEPNP_EPNP);
    if (!success || inlierIndices.size() < static_cast<std::size_t>(minInliers_)) {
        return std::nullopt;
    }

    cv::Mat rotationMatrix;
    cv::Rodrigues(rotationVector, rotationMatrix);
    cv::Mat rotation64;
    cv::Mat translation64;
    rotationMatrix.convertTo(rotation64, CV_64F);
    translation.convertTo(translation64, CV_64F);
    cv::Matx33d cameraWorld;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            cameraWorld(row, column) = rotation64.at<double>(row, column);
        }
    }
    const cv::Vec3d cameraTranslation(translation64.at<double>(0),
                                     translation64.at<double>(1),
                                     translation64.at<double>(2));
    rotationWorldCamera_ = cameraWorld.t();
    positionWorld_ = -(rotationWorldCamera_ * cameraTranslation);

    std::vector<cv::Point3f> inlierWorld;
    std::vector<cv::Point2f> inlierObservations;
    inlierWorld.reserve(inlierIndices.size());
    inlierObservations.reserve(inlierIndices.size());
    for (int index : inlierIndices) {
        inlierWorld.push_back(worldPoints[static_cast<std::size_t>(index)]);
        inlierObservations.push_back(observations[static_cast<std::size_t>(index)]);
    }
    std::vector<cv::Point2f> projected;
    cv::projectPoints(inlierWorld, rotationVector, translation, cameraMatrix(camera_),
                      cv::noArray(), projected);
    std::vector<double> errors;
    errors.reserve(inlierIndices.size());
    for (std::size_t i = 0; i < projected.size(); ++i) {
        errors.push_back(cv::norm(projected[i] - inlierObservations[i]));
    }
    return PoseEstimate{
        static_cast<double>(inlierIndices.size()) /
            static_cast<double>(std::max<std::size_t>(1, imagePoints.size())),
        median(std::move(errors)), static_cast<int>(inlierIndices.size())};
}

void FeatureSLAM::triangulateNewLandmarks(
    const std::vector<cv::Point2f>& oldPoints,
    const std::vector<cv::Point2f>& newPoints, std::vector<int>& landmarkIds,
    const cv::Matx33d& previousRotation, const cv::Vec3d& previousPosition) {
    if (oldPoints.size() != newPoints.size() ||
        newPoints.size() != landmarkIds.size()) {
        return;
    }
    std::vector<cv::Point2f> oldUnmapped;
    std::vector<cv::Point2f> newUnmapped;
    std::vector<std::size_t> originalIndices;
    for (std::size_t i = 0; i < landmarkIds.size(); ++i) {
        if (landmarkIds[i] < 0) {
            oldUnmapped.push_back(oldPoints[i]);
            newUnmapped.push_back(newPoints[i]);
            originalIndices.push_back(i);
        }
    }
    if (oldUnmapped.size() < 5) {
        return;
    }

    const cv::Matx33d previousCameraRotation = previousRotation.t();
    const cv::Matx33d currentCameraRotation = rotationWorldCamera_.t();
    const cv::Vec3d previousTranslation =
        -(previousCameraRotation * previousPosition);
    const cv::Vec3d currentTranslation = -(currentCameraRotation * positionWorld_);
    const cv::Mat projectionPrevious =
        projectionMatrix(camera_, previousCameraRotation, previousTranslation);
    const cv::Mat projectionCurrent =
        projectionMatrix(camera_, currentCameraRotation, currentTranslation);

    cv::Mat homogeneous;
    cv::triangulatePoints(projectionPrevious, projectionCurrent, oldUnmapped,
                          newUnmapped, homogeneous);
    cv::Mat homogeneous64;
    homogeneous.convertTo(homogeneous64, CV_64F);

    for (std::size_t i = 0; i < oldUnmapped.size(); ++i) {
        const double w = homogeneous64.at<double>(3, static_cast<int>(i));
        if (std::abs(w) < 1e-9) {
            continue;
        }
        const cv::Vec3d point(homogeneous64.at<double>(0, static_cast<int>(i)) / w,
                              homogeneous64.at<double>(1, static_cast<int>(i)) / w,
                              homogeneous64.at<double>(2, static_cast<int>(i)) / w);
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
            !std::isfinite(point[2])) {
            continue;
        }
        const cv::Vec3d pointPrevious =
            previousCameraRotation * point + previousTranslation;
        const cv::Vec3d pointCurrent =
            currentCameraRotation * point + currentTranslation;
        if (pointPrevious[2] <= 1e-6 || pointCurrent[2] <= 1e-6) {
            continue;
        }
        const cv::Point2d previousPixel(
            camera_.fx * pointPrevious[0] / pointPrevious[2] + camera_.cx,
            camera_.fy * pointPrevious[1] / pointPrevious[2] + camera_.cy);
        const cv::Point2d currentPixel(
            camera_.fx * pointCurrent[0] / pointCurrent[2] + camera_.cx,
            camera_.fy * pointCurrent[1] / pointCurrent[2] + camera_.cy);
        if (cv::norm(previousPixel - cv::Point2d(oldUnmapped[i])) >= 3.0 ||
            cv::norm(currentPixel - cv::Point2d(newUnmapped[i])) >= 3.0) {
            continue;
        }
        const cv::Vec3d previousRay =
            previousRotation *
            cv::Vec3d((oldUnmapped[i].x - camera_.cx) / camera_.fx,
                      (oldUnmapped[i].y - camera_.cy) / camera_.fy, 1.0);
        const cv::Vec3d currentRay =
            rotationWorldCamera_ *
            cv::Vec3d((newUnmapped[i].x - camera_.cx) / camera_.fx,
                      (newUnmapped[i].y - camera_.cy) / camera_.fy, 1.0);
        const double cosine =
            previousRay.dot(currentRay) /
            (cv::norm(previousRay) * cv::norm(currentRay));
        if (cosine >= std::cos(CV_PI / 360.0)) {
            continue;
        }
        landmarkIds[originalIndices[i]] = static_cast<int>(landmarks_.size());
        landmarks_.emplace_back(point[0], point[1], point[2]);
    }
}

double FeatureSLAM::confidenceScore(double trackedRatio, double inlierRatio,
                                    double reprojectionErrorPx) {
    const double geometricQuality =
        std::isfinite(reprojectionErrorPx)
            ? std::exp(-std::max(0.0, reprojectionErrorPx) / 2.0)
            : 0.0;
    return std::clamp(0.35 * trackedRatio + 0.45 * inlierRatio +
                          0.20 * geometricQuality,
                      0.0, 1.0);
}

const std::vector<cv::Point3d>& FeatureSLAM::landmarks() const noexcept {
    return landmarks_;
}

const std::vector<Keyframe>& FeatureSLAM::keyframes() const noexcept {
    return keyframes_;
}

}  // namespace adaptive_slam
