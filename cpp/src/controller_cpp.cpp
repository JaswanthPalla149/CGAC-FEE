#include "adaptive_slam/slam.hpp"

#include <stdexcept>

namespace adaptive_slam {

AdaptiveController::AdaptiveController(double enterRecoveryBelow,
                                       double exitRecoveryAbove) {
    if (adaptive_slam_controller_init(&controller_, enterRecoveryBelow,
                                      exitRecoveryAbove) != 0) {
        throw std::invalid_argument(
            "thresholds must satisfy 0 <= enter < exit <= 1");
    }
}

TrackingMode AdaptiveController::choose(double confidence,
                                        TrackingMode currentMode) const {
    const adaptive_slam_mode current =
        currentMode == TrackingMode::Recovery ? ADAPTIVE_SLAM_RECOVERY
                                               : ADAPTIVE_SLAM_LOW_COST;
    adaptive_slam_mode next = ADAPTIVE_SLAM_LOW_COST;
    if (adaptive_slam_controller_choose(&controller_, confidence, current,
                                        &next) != 0) {
        throw std::invalid_argument("confidence must be between 0 and 1");
    }
    return next == ADAPTIVE_SLAM_RECOVERY ? TrackingMode::Recovery
                                          : TrackingMode::LowCost;
}

}  // namespace adaptive_slam
