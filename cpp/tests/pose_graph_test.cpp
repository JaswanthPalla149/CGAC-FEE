#include "adaptive_slam/pose_graph.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

double distance(const cv::Vec3d& first, const cv::Vec3d& second) {
    return cv::norm(first - second);
}

}  // namespace

int main() {
    using adaptive_slam::PoseGraphPose;
    using adaptive_slam::makePoseGraphConstraint;
    using adaptive_slam::optimizePoseGraph;

    const std::vector<PoseGraphPose> truth{
        {cv::Matx33d::eye(), {0.0, 0.0, 0.0}},
        {cv::Matx33d::eye(), {1.0, 0.0, 0.0}},
        {cv::Matx33d::eye(), {1.0, 1.0, 0.0}},
        {cv::Matx33d::eye(), {0.0, 1.0, 0.0}},
        {cv::Matx33d::eye(), {0.0, 0.0, 0.0}},
    };
    std::vector<PoseGraphPose> estimate = truth;
    estimate[1].positionWorld += cv::Vec3d(0.08, 0.03, 0.0);
    estimate[2].positionWorld += cv::Vec3d(0.14, 0.09, 0.02);
    estimate[3].positionWorld += cv::Vec3d(0.18, 0.12, -0.01);
    estimate[4].positionWorld = {35.0, 20.0, 8.0};
    const double initialClosureError =
        distance(estimate.front().positionWorld, estimate.back().positionWorld);

    std::vector<adaptive_slam::PoseGraphConstraint> constraints;
    for (std::size_t node = 0; node + 1 < truth.size(); ++node) {
        constraints.push_back(makePoseGraphConstraint(
            node, node + 1, truth[node], truth[node + 1], 100.0, 100.0));
    }
    constraints.push_back(makePoseGraphConstraint(
        0, truth.size() - 1, truth.front(), truth.back(), 100.0, 100.0, true));

    assert(optimizePoseGraph(estimate, constraints, 12));
    const double finalClosureError =
        distance(estimate.front().positionWorld, estimate.back().positionWorld);
    assert(finalClosureError < initialClosureError);
    assert(finalClosureError < 1e-3);

    std::vector<PoseGraphPose> invalidGraph = truth;
    auto invalidConstraints = constraints;
    invalidConstraints.back().to = truth.size();
    assert(!optimizePoseGraph(invalidGraph, invalidConstraints));
    return 0;
}
