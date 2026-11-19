#include "adaptive_slam/slam.hpp"

#include <cmath>
#include <stdexcept>

namespace adaptive_slam {

AdaptiveController::AdaptiveController(double enterRecoveryBelow,
                                       double exitRecoveryAbove,
                                       double enterVisualInertialBelow,
                                       double exitVisualInertialAbove)
    : enterVisualInertialBelow_(enterVisualInertialBelow),
      exitVisualInertialAbove_(exitVisualInertialAbove) {
    if (adaptive_slam_controller_init(&controller_, enterRecoveryBelow,
                                      exitRecoveryAbove) != 0 ||
        !std::isfinite(enterVisualInertialBelow_) ||
        !std::isfinite(exitVisualInertialAbove_) ||
        enterVisualInertialBelow_ < 0.0 ||
        enterVisualInertialBelow_ >= enterRecoveryBelow ||
        exitVisualInertialAbove_ < enterRecoveryBelow ||
        exitVisualInertialAbove_ > 1.0 ||
        enterVisualInertialBelow_ >= exitVisualInertialAbove_) {
        throw std::invalid_argument(
            "thresholds must satisfy 0 <= enter visual-inertial < enter "
            "recovery <= exit visual-inertial <= 1 and enter recovery < "
            "exit recovery");
    }
}

TrackingMode AdaptiveController::choose(double confidence,
                                        TrackingMode currentMode,
                                        bool imuAvailable) const {
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
        throw std::invalid_argument("confidence must be between 0 and 1");
    }
    if (currentMode != TrackingMode::LowCost &&
        currentMode != TrackingMode::Recovery &&
        currentMode != TrackingMode::VisualInertial) {
        throw std::invalid_argument("invalid current tracking mode");
    }
    if (currentMode == TrackingMode::VisualInertial && imuAvailable &&
        confidence <= exitVisualInertialAbove_) {
        return TrackingMode::VisualInertial;
    }
    const adaptive_slam_mode current =
        currentMode == TrackingMode::LowCost ? ADAPTIVE_SLAM_LOW_COST
                                              : ADAPTIVE_SLAM_RECOVERY;
    adaptive_slam_mode next = ADAPTIVE_SLAM_LOW_COST;
    if (adaptive_slam_controller_choose(&controller_, confidence, current,
                                        &next) != 0) {
        throw std::invalid_argument("confidence must be between 0 and 1");
    }
    if (imuAvailable && confidence < enterVisualInertialBelow_) {
        return TrackingMode::VisualInertial;
    }
    return next == ADAPTIVE_SLAM_RECOVERY ? TrackingMode::Recovery
                                          : TrackingMode::LowCost;
}

}  // namespace adaptive_slam
