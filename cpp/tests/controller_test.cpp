#include "adaptive_slam/slam.hpp"

#include <stdexcept>

int main() {
    using adaptive_slam::AdaptiveController;
    using adaptive_slam::TrackingMode;

    const AdaptiveController controller;
    if (controller.choose(0.3, TrackingMode::LowCost) !=
            TrackingMode::Recovery ||
        controller.choose(0.55, TrackingMode::Recovery) !=
            TrackingMode::Recovery ||
        controller.choose(0.59, TrackingMode::Recovery) !=
            TrackingMode::Recovery ||
        controller.choose(0.61, TrackingMode::Recovery) !=
            TrackingMode::LowCost ||
        controller.choose(0.7, TrackingMode::Recovery) !=
            TrackingMode::LowCost ||
        controller.choose(0.55, TrackingMode::LowCost) !=
            TrackingMode::LowCost) {
        return 1;
    }
    if (controller.choose(0.24, TrackingMode::LowCost, true) !=
            TrackingMode::VisualInertial ||
        controller.choose(0.40, TrackingMode::VisualInertial, true) !=
            TrackingMode::VisualInertial ||
        controller.choose(0.50, TrackingMode::VisualInertial, true) !=
            TrackingMode::Recovery ||
        controller.choose(0.20, TrackingMode::VisualInertial, false) !=
            TrackingMode::Recovery) {
        return 3;
    }

    bool invalidConfidenceRejected = false;
    try {
        controller.choose(1.1, TrackingMode::LowCost);
    } catch (const std::invalid_argument&) {
        invalidConfidenceRejected = true;
    }
    bool invalidModeRejected = false;
    try {
        controller.choose(0.5, static_cast<TrackingMode>(99), true);
    } catch (const std::invalid_argument&) {
        invalidModeRejected = true;
    }
    return invalidConfidenceRejected && invalidModeRejected ? 0 : 2;
}
