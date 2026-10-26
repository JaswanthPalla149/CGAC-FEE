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
        controller.choose(0.7, TrackingMode::Recovery) !=
            TrackingMode::LowCost ||
        controller.choose(0.55, TrackingMode::LowCost) !=
            TrackingMode::LowCost) {
        return 1;
    }

    bool invalidConfidenceRejected = false;
    try {
        controller.choose(1.1, TrackingMode::LowCost);
    } catch (const std::invalid_argument&) {
        invalidConfidenceRejected = true;
    }
    return invalidConfidenceRejected ? 0 : 2;
}
