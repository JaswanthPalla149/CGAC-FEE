#include "adaptive_slam/controller.h"

int main(void) {
    adaptive_slam_controller controller;
    adaptive_slam_mode mode = ADAPTIVE_SLAM_LOW_COST;
    if (adaptive_slam_controller_init(&controller, 0.48, 0.68) != 0) {
        return 1;
    }
    if (adaptive_slam_controller_choose(&controller, 0.3,
                                        ADAPTIVE_SLAM_LOW_COST, &mode) != 0 ||
        mode != ADAPTIVE_SLAM_RECOVERY) {
        return 2;
    }
    if (adaptive_slam_controller_choose(&controller, 0.55,
                                        ADAPTIVE_SLAM_RECOVERY, &mode) != 0 ||
        mode != ADAPTIVE_SLAM_RECOVERY) {
        return 3;
    }
    if (adaptive_slam_controller_choose(&controller, 0.7,
                                        ADAPTIVE_SLAM_RECOVERY, &mode) != 0 ||
        mode != ADAPTIVE_SLAM_LOW_COST) {
        return 4;
    }
    if (adaptive_slam_controller_choose(&controller, 1.1,
                                        ADAPTIVE_SLAM_LOW_COST, &mode) == 0) {
        return 5;
    }
    return 0;
}
