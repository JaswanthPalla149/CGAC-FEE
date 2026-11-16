#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace adaptive_slam {

struct PoseGraphPose {
    cv::Matx33d rotationWorldFromBody{cv::Matx33d::eye()};
    cv::Vec3d positionWorld{0.0, 0.0, 0.0};
};

struct PoseGraphConstraint {
    std::size_t from{};
    std::size_t to{};
    cv::Matx33d rotationFromTo{cv::Matx33d::eye()};
    cv::Vec3d translationFromTo{0.0, 0.0, 0.0};
    double translationInformation{1.0};
    double rotationInformation{1.0};
    bool loopClosure{false};
};

PoseGraphConstraint makePoseGraphConstraint(
    std::size_t from, std::size_t to, const PoseGraphPose& poseFrom,
    const PoseGraphPose& poseTo, double translationInformation,
    double rotationInformation, bool loopClosure = false);

bool optimizePoseGraph(std::vector<PoseGraphPose>& poses,
                       const std::vector<PoseGraphConstraint>& constraints,
                       int maxIterations = 12);

}  // namespace adaptive_slam
