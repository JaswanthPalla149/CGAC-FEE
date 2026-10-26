#ifndef ADAPTIVE_SLAM_CONTROLLER_H
#define ADAPTIVE_SLAM_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum adaptive_slam_mode {
    ADAPTIVE_SLAM_LOW_COST = 0,
    ADAPTIVE_SLAM_RECOVERY = 1
} adaptive_slam_mode;

typedef struct adaptive_slam_controller {
    double enter_recovery_below;
    double exit_recovery_above;
} adaptive_slam_controller;

int adaptive_slam_controller_init(adaptive_slam_controller* controller,
                                  double enter_recovery_below,
                                  double exit_recovery_above);
int adaptive_slam_controller_choose(const adaptive_slam_controller* controller,
                                    double confidence,
                                    adaptive_slam_mode current_mode,
                                    adaptive_slam_mode* next_mode);

#ifdef __cplusplus
}
#endif

#endif
