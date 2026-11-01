#include "adaptive_slam/controller.h"

#include <math.h>

int adaptive_slam_controller_init(adaptive_slam_controller* controller,
                                  double enter_recovery_below,
                                  double exit_recovery_above) {
    if (controller == 0 || !isfinite(enter_recovery_below) ||
        !isfinite(exit_recovery_above) || enter_recovery_below < 0.0 ||
        enter_recovery_below >= exit_recovery_above ||
        exit_recovery_above > 1.0) {
        return 1;
    }
    controller->enter_recovery_below = enter_recovery_below;
    controller->exit_recovery_above = exit_recovery_above;
    return 0;
}

int adaptive_slam_controller_choose(const adaptive_slam_controller* controller,
                                    double confidence,
                                    adaptive_slam_mode current_mode,
                                    adaptive_slam_mode* next_mode) {
    if (controller == 0 || next_mode == 0 || !isfinite(confidence) ||
        confidence < 0.0 || confidence > 1.0 ||
        (current_mode != ADAPTIVE_SLAM_LOW_COST &&
         current_mode != ADAPTIVE_SLAM_RECOVERY)) {
        return 1;
    }
    if (current_mode == ADAPTIVE_SLAM_LOW_COST) {
        *next_mode = confidence < controller->enter_recovery_below
                         ? ADAPTIVE_SLAM_RECOVERY
                         : ADAPTIVE_SLAM_LOW_COST;
    } else {
        *next_mode = confidence > controller->exit_recovery_above
                         ? ADAPTIVE_SLAM_LOW_COST
                         : ADAPTIVE_SLAM_RECOVERY;
    }
    return 0;
}
