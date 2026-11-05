#include "adaptive_slam/slam.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdlib>

namespace {

// Livox Avia multi-modal raycasting against 3D Retail Street environment
adaptive_slam::LidarScan simulateAviaScan(
    int64_t timestampNs,
    const cv::Vec3d& uavPos,
    const cv::Matx33d& uavRot,
    int numPoints = 250) {
    adaptive_slam::LidarScan scan;
    scan.timestampNs = timestampNs;
    scan.points.reserve(numPoints);

    for (int p = 0; p < numPoints; ++p) {
        // Avia rotating prism rosette equations
        const double t = static_cast<double>(p) * (2.0 * 3.141592653589793 / 100.0);
        const double fov_az = 0.55 * std::sin(t * 1.7) * std::cos(t * 0.3);
        const double fov_el = 0.45 * std::cos(t * 2.3) + 0.15 * std::sin(t * 0.7);

        // Ray unit direction in UAV body frame (forward +Y, right +X, up +Z)
        cv::Vec3d rayBody(std::sin(fov_az), std::cos(fov_az) * std::cos(fov_el), std::sin(fov_el));
        const double rayNorm = cv::norm(rayBody);
        rayBody = rayBody / (rayNorm > 1e-6 ? rayNorm : 1.0);

        // Ray in world coordinates
        cv::Vec3d rayWorld = uavRot * rayBody;

        double minRange = 100.0;
        bool hit = false;

        // 1. Storefront facade intersection at Y_world = 3.8 m (Burger King / Nayuki)
        if (std::abs(rayWorld[1]) > 1e-4) {
            const double s = (3.8 - uavPos[1]) / rayWorld[1];
            if (s > 0.3 && s < minRange) {
                const double hitX = uavPos[0] + s * rayWorld[0];
                const double hitZ = uavPos[2] + s * rayWorld[2];
                if (hitX >= -8.0 && hitX <= 8.0 && hitZ >= 0.0 && hitZ <= 4.0) {
                    minRange = s;
                    hit = true;
                }
            }
        }

        // 2. Ground / Pavement plane intersection at Z_world = 0.0 m
        if (std::abs(rayWorld[2]) > 1e-4) {
            const double s = -uavPos[2] / rayWorld[2];
            if (s > 0.3 && s < minRange) {
                const double hitX = uavPos[0] + s * rayWorld[0];
                const double hitY = uavPos[1] + s * rayWorld[1];
                if (hitX >= -8.0 && hitX <= 8.0 && hitY >= -2.0 && hitY <= 6.0) {
                    minRange = s;
                    hit = true;
                }
            }
        }

        // 3. Side wall & pillar intersections at X_world = +/- 4.5 m
        if (std::abs(rayWorld[0]) > 1e-4) {
            for (double wallX : {-4.5, 4.5}) {
                const double s = (wallX - uavPos[0]) / rayWorld[0];
                if (s > 0.3 && s < minRange) {
                    const double hitY = uavPos[1] + s * rayWorld[1];
                    const double hitZ = uavPos[2] + s * rayWorld[2];
                    if (hitY >= -2.0 && hitY <= 6.0 && hitZ >= 0.0 && hitZ <= 4.0) {
                        minRange = s;
                        hit = true;
                    }
                }
            }
        }

        if (hit && minRange < 35.0) {
            cv::Vec3d measBody = rayBody * minRange;
            scan.points.push_back({
                static_cast<float>(measBody[0]),
                static_cast<float>(measBody[1]),
                static_cast<float>(measBody[2]),
                100.0f
            });
        }
    }
    return scan;
}

// RViz-style 3D Visualizer Canvas (matching FAST-LIVO2 RViz screenshot)
void renderFastLivo2Simulation(
    cv::Mat& canvas,
    const cv::Mat& currentImage,
    const std::vector<cv::Point2f>& trackedFeatures,
    const std::vector<cv::Point3d>& mapPoints,
    const std::vector<cv::Point3d>& currentScanWorld,
    const std::vector<adaptive_slam::UAVState>& trajectory,
    const adaptive_slam::UAVState& state,
    int currentFrame,
    int totalFrames,
    double currentAte) {

    canvas = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(18, 18, 18)); // RViz dark background

    // ==========================================
    // 1. LEFT PANE (Width 400, Height 720): Displays & Camera
    // ==========================================
    cv::rectangle(canvas, cv::Rect(0, 0, 420, 720), cv::Scalar(32, 32, 32), -1);
    cv::line(canvas, cv::Point(420, 0), cv::Point(420, 720), cv::Scalar(65, 65, 65), 2);

    // Title Header
    cv::rectangle(canvas, cv::Rect(0, 0, 420, 36), cv::Scalar(48, 48, 48), -1);
    cv::putText(canvas, "fast_livo2.rviz* - RViz (EcoLIVO)", cv::Point(12, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(240, 240, 240), 2);

    // Displays Tree
    int y = 60;
    cv::putText(canvas, "Displays", cv::Point(14, y), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(140, 200, 255), 1);
    y += 24;
    cv::putText(canvas, "[v] Global Status: Ok", cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 255, 100), 1);
    y += 20;
    cv::putText(canvas, "    Fixed Frame: map", cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(200, 200, 200), 1);
    y += 20;
    cv::putText(canvas, "[v] Grid: Ok (Fixed Frame)", cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 255, 100), 1);
    y += 20;
    cv::putText(canvas, "[v] Odometry: /state_estimation/odometry", cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 255, 100), 1);
    y += 20;
    cv::putText(canvas, "[v] PointCloud2: /cloud_registered", cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 255, 100), 1);
    y += 18;
    std::string mapPtsStr = "    Points: " + std::to_string(mapPoints.size()) + " registered";
    cv::putText(canvas, mapPtsStr, cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(180, 180, 180), 1);
    y += 20;
    cv::putText(canvas, "[v] Image: /camera/image_raw", cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 255, 100), 1);

    // SLAM Status & Telemetry Box
    y += 28;
    cv::rectangle(canvas, cv::Rect(14, y, 390, 140), cv::Scalar(22, 22, 22), -1);
    cv::rectangle(canvas, cv::Rect(14, y, 390, 140), cv::Scalar(75, 75, 75), 1);

    int ty = y + 22;
    std::string frameStr = "Frame: " + std::to_string(currentFrame) + " / " + std::to_string(totalFrames);
    cv::putText(canvas, frameStr, cv::Point(24, ty), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 255, 255), 1);

    // Power mode badge
    std::string modeStr = adaptive_slam::hybridModeName(state.mode);
    cv::Scalar badgeColor(0, 200, 0); // Green for low_cost
    if (state.mode == adaptive_slam::HybridPowerMode::Recovery) badgeColor = cv::Scalar(0, 165, 255); // Orange
    else if (state.mode == adaptive_slam::HybridPowerMode::VisualBypassLIO) badgeColor = cv::Scalar(220, 50, 220); // Magenta

    cv::putText(canvas, "Mode: " + modeStr, cv::Point(190, ty), cv::FONT_HERSHEY_DUPLEX, 0.50, badgeColor, 1);

    ty += 22;
    std::stringstream confStream;
    confStream << std::fixed << std::setprecision(3) << "Confidence: " << state.confidence
               << " (V:" << state.visualInliers << " L:" << state.lidarInliers << ")";
    cv::putText(canvas, confStream.str(), cv::Point(24, ty), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(220, 220, 220), 1);

    ty += 22;
    std::stringstream posStream;
    posStream << std::fixed << std::setprecision(2) << "Pose: (" << state.positionWorld[0]
              << ", " << state.positionWorld[1] << ", " << state.positionWorld[2] << ") m";
    cv::putText(canvas, posStream.str(), cv::Point(24, ty), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(100, 220, 255), 1);

    ty += 22;
    std::stringstream ateStream;
    ateStream << std::fixed << std::setprecision(3) << "ATE Error: " << currentAte << " m | "
              << state.elapsedMs << " ms";
    cv::putText(canvas, ateStream.str(), cv::Point(24, ty), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(150, 255, 150), 1);

    ty += 22;
    std::stringstream macStream;
    macStream << "MACs: " << state.estimatedMacCount << " | Energy: "
              << std::fixed << std::setprecision(4) << state.estimatedEnergyJoules << " J";
    cv::putText(canvas, macStream.str(), cv::Point(24, ty), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 210, 100), 1);

    // Live Camera Sub-view (Bottom-Left)
    int camTop = 385;
    cv::rectangle(canvas, cv::Rect(14, camTop, 392, 320), cv::Scalar(10, 10, 10), -1);
    cv::rectangle(canvas, cv::Rect(14, camTop, 392, 320), cv::Scalar(80, 80, 80), 2);
    cv::putText(canvas, "Camera [/camera/image_raw]", cv::Point(20, camTop - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

    if (!currentImage.empty()) {
        cv::Mat displayImg;
        if (currentImage.channels() == 1) {
            cv::cvtColor(currentImage, displayImg, cv::COLOR_GRAY2BGR);
        } else {
            displayImg = currentImage.clone();
        }

        // Overlay tracked feature points (green for tracked, blue for new)
        for (std::size_t i = 0; i < trackedFeatures.size(); ++i) {
            const auto& pt = trackedFeatures[i];
            if (pt.x >= 0 && pt.x < displayImg.cols && pt.y >= 0 && pt.y < displayImg.rows) {
                if (i % 2 == 0) {
                    cv::circle(displayImg, pt, 3, cv::Scalar(0, 255, 0), -1); // Green
                } else {
                    cv::circle(displayImg, pt, 3, cv::Scalar(255, 120, 0), -1); // Cyan/Blue
                }
            }
        }

        cv::Mat resizedCam;
        cv::resize(displayImg, resizedCam, cv::Size(390, 316));
        resizedCam.copyTo(canvas(cv::Rect(15, camTop + 2, 390, 316)));
    }

    // ==========================================
    // 2. RIGHT PANE (Width 860, Height 720): 3D Viewport
    // ==========================================
    const cv::Rect vp(422, 0, 856, 720);

    // Virtual 3D perspective projection parameters
    // Viewing the street from elevated perspective: center around (0, 2.0, 1.0)
    const double viewCenterX = 0.0;
    const double viewCenterY = 1.8;
    const double viewCenterZ = 0.8;
    const double scale3D = 75.0; // Pixels per meter

    auto project3D = [&](double x, double y, double z) -> cv::Point {
        // Isometric-perspective tilt
        const double dx = x - viewCenterX;
        const double dy = y - viewCenterY;
        const double dz = z - viewCenterZ;

        // Rotate ~30 deg elevation and ~20 deg azimuth
        const double screenX = vp.x + vp.width * 0.5 + (dx * 0.88 - dy * 0.45) * scale3D;
        const double screenY = vp.y + vp.height * 0.58 - (dz * 0.85 + dy * 0.35 + dx * 0.15) * scale3D;
        return cv::Point(static_cast<int>(screenX), static_cast<int>(screenY));
    };

    // Draw Ground Grid on Z = 0
    for (double gx = -6.0; gx <= 6.0; gx += 1.0) {
        cv::Point p1 = project3D(gx, -1.0, 0.0);
        cv::Point p2 = project3D(gx, 5.0, 0.0);
        if (vp.contains(p1) || vp.contains(p2)) {
            cv::line(canvas, p1, p2, cv::Scalar(38, 38, 38), 1);
        }
    }
    for (double gy = -1.0; gy <= 5.0; gy += 1.0) {
        cv::Point p1 = project3D(-6.0, gy, 0.0);
        cv::Point p2 = project3D(6.0, gy, 0.0);
        if (vp.contains(p1) || vp.contains(p2)) {
            cv::line(canvas, p1, p2, cv::Scalar(38, 38, 38), 1);
        }
    }

    // Origin Axes (Red: X, Green: Y, Blue: Z)
    cv::Point o = project3D(0.0, 0.0, 0.0);
    cv::Point axX = project3D(1.0, 0.0, 0.0);
    cv::Point axY = project3D(0.0, 1.0, 0.0);
    cv::Point axZ = project3D(0.0, 0.0, 1.0);
    cv::line(canvas, o, axX, cv::Scalar(0, 0, 255), 2); // X red
    cv::line(canvas, o, axY, cv::Scalar(0, 255, 0), 2); // Y green
    cv::line(canvas, o, axZ, cv::Scalar(255, 0, 0), 2); // Z blue

    // Storefront Outline Landmark
    cv::Point w1 = project3D(-4.5, 3.8, 0.0);
    cv::Point w2 = project3D(4.5, 3.8, 0.0);
    cv::Point w3 = project3D(4.5, 3.8, 2.8);
    cv::Point w4 = project3D(-4.5, 3.8, 2.8);
    cv::line(canvas, w1, w2, cv::Scalar(70, 70, 70), 1);
    cv::line(canvas, w2, w3, cv::Scalar(70, 70, 70), 1);
    cv::line(canvas, w3, w4, cv::Scalar(70, 70, 70), 1);
    cv::line(canvas, w4, w1, cv::Scalar(70, 70, 70), 1);
    cv::putText(canvas, "RETAIL FACADE [BURGER KING]", project3D(-2.5, 3.8, 2.9),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(160, 160, 160), 1);

    // Render Accumulated 3D Registered Map (/cloud_registered)
    // Subsample if large for fast 60+ FPS rendering
    const std::size_t mapStep = std::max<std::size_t>(1, mapPoints.size() / 1500);
    for (std::size_t i = 0; i < mapPoints.size(); i += mapStep) {
        const auto& pt = mapPoints[i];
        cv::Point scr = project3D(pt.x, pt.y, pt.z);
        if (vp.contains(scr)) {
            // Elevation colormap
            const double normZ = std::clamp((pt.z - 0.0) / 2.8, 0.0, 1.0);
            const int r = static_cast<int>(255 * normZ);
            const int g = static_cast<int>(255 * (1.0 - std::abs(normZ - 0.5) * 2.0));
            const int b = static_cast<int>(255 * (1.0 - normZ));
            cv::circle(canvas, scr, 1, cv::Scalar(b, g, r), -1);
        }
    }

    // Render Current LiDAR Scan (/lidar/points) in BRIGHT RED (as in fast_livo2 RViz)
    for (const auto& pt : currentScanWorld) {
        cv::Point scr = project3D(pt.x, pt.y, pt.z);
        if (vp.contains(scr)) {
            cv::circle(canvas, scr, 2, cv::Scalar(0, 0, 255), -1); // Bright Red
        }
    }

    // Render 3D Trajectory Path (Yellow-Green line)
    if (trajectory.size() >= 2) {
        for (std::size_t i = 1; i < trajectory.size(); ++i) {
            const auto& pA = trajectory[i - 1].positionWorld;
            const auto& pB = trajectory[i].positionWorld;
            cv::Point scrA = project3D(pA[0], pA[1], pA[2]);
            cv::Point scrB = project3D(pB[0], pB[1], pB[2]);
            if (vp.contains(scrA) && vp.contains(scrB)) {
                cv::line(canvas, scrA, scrB, cv::Scalar(0, 255, 220), 2);
            }
        }
    }

    // Render Current UAV Pose Body
    cv::Point uavScreen = project3D(state.positionWorld[0], state.positionWorld[1], state.positionWorld[2]);
    if (vp.contains(uavScreen)) {
        cv::circle(canvas, uavScreen, 6, cv::Scalar(0, 255, 255), -1); // Yellow dot
        cv::circle(canvas, uavScreen, 9, cv::Scalar(255, 255, 255), 1);
        // Body direction arrow
        cv::Vec3d fwd = state.rotationWorldBody * cv::Vec3d(0.0, 0.5, 0.0);
        cv::Point fwdScr = project3D(state.positionWorld[0] + fwd[0],
                                     state.positionWorld[1] + fwd[1],
                                     state.positionWorld[2] + fwd[2]);
        cv::arrowedLine(canvas, uavScreen, fwdScr, cv::Scalar(0, 255, 0), 2, 8, 0, 0.3);
    }

    // Top Legend Header
    cv::rectangle(canvas, cv::Rect(430, 10, 520, 28), cv::Scalar(28, 28, 28), -1);
    cv::putText(canvas, "Topic: /cloud_registered (Style: Points)  |  Red: Current Avia Scan",
                cv::Point(440, 30), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(230, 230, 230), 1);
}

std::vector<std::string> splitList(const std::string& text) {
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string value;
    while (std::getline(stream, value, ',')) {
        if (!value.empty()) {
            values.push_back(value);
        }
    }
    return values;
}

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " --sequence PATH [--output PATH] [--degradations LIST] [--threads N] [--uav-hybrid] [--test-shock SHOCK] [--fps N] [--gui]\n"
        << "  LIST: comma-separated none,blur,low_light,noise\n"
        << "  SHOCK: blackout, solar_flare, strobe\n"
        << "  Example: " << executable
        << " --sequence synthetic/ecolivo-wsl --output results/ecolivo --uav-hybrid --fps 10\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path sequencePath;
        std::filesystem::path outputPath{"results"};
        std::vector<std::string> degradations{"none"};
        int threads = 1;
        bool runUavHybrid = false;
        bool enableGui = false;
        std::string testShock;
        double replayFps = 0.0;

        for (int i = 1; i < argc; ++i) {
            std::string argument = argv[i];
            const std::size_t argumentFirst = argument.find_first_not_of(" \t\r\n");
            if (argumentFirst == std::string::npos) continue;
            argument = argument.substr(argumentFirst);
            if (argument == "\\") continue;

            if (argument == "--help" || argument == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            if (argument == "--uav-hybrid") {
                runUavHybrid = true;
                continue;
            }
            if (argument == "--gui") {
                enableGui = true;
                continue;
            }
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            std::string value = argv[++i];
            const std::size_t valueFirst = value.find_first_not_of(" \t\r\n");
            const std::size_t valueLast = value.find_last_not_of(" \t\r\n");
            if (valueFirst != std::string::npos) {
                value = value.substr(valueFirst, valueLast - valueFirst + 1);
            } else {
                value.clear();
            }

            if (argument == "--sequence") {
                sequencePath = value;
            } else if (argument == "--output") {
                outputPath = value;
            } else if (argument == "--degradations") {
                degradations = splitList(value);
            } else if (argument == "--threads") {
                threads = std::stoi(value);
            } else if (argument == "--test-shock") {
                testShock = value;
                runUavHybrid = true;
            } else if (argument == "--fps") {
                replayFps = std::stod(value);
            } else {
                throw std::invalid_argument("unknown argument: " + argument);
            }
        }

        if (sequencePath.empty()) {
            printUsage(argv[0]);
            return 2;
        }
        if (threads < 1) {
            throw std::invalid_argument("--threads must be a positive integer");
        }
        if (replayFps < 0.0 || replayFps > 240.0) {
            throw std::invalid_argument("--fps must be between 0 and 240");
        }

        cv::setUseOptimized(true);
        cv::setNumThreads(threads);

        if (runUavHybrid) {
            std::cout << "\n=======================================================\n"
                      << "Running 3D UAV Hybrid LiDAR-Visual-Inertial SLAM (EcoLIVO)\n"
                      << "=======================================================\n";
            const auto sequence = adaptive_slam::loadEuRoC(sequencePath);
            adaptive_slam::HybridSlamConfig config;
            config.enableIntegerClahe = true;
            config.enableHbstMatching = true;
            adaptive_slam::UAVHybridSLAM hybridSlam(sequence.camera, config);
            if (sequence.groundTruth.has_value() && !sequence.groundTruth->positions.empty()) {
                hybridSlam.reset(sequence.groundTruth->positions[0]);
            }

            std::filesystem::create_directories(outputPath);

            // Check if GUI window should be displayed (WSLg, Linux X11, or Windows)
            const bool hasDisplay = (std::getenv("DISPLAY") != nullptr) || (std::getenv("WAYLAND_DISPLAY") != nullptr);
            const bool showLiveGui = enableGui || hasDisplay;

            std::string windowName = "fast_livo2.rviz* - EcoLIVO Simulation";
            if (showLiveGui) {
                try {
                    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                } catch (...) {
                    // Headless fallback
                }
            }

            int lowCostFrames = 0;
            int recoveryFrames = 0;
            int bypassFrames = 0;
            const std::size_t printInterval = std::max<std::size_t>(1, sequence.frames.size() / 30);

            cv::Mat rvizCanvas;

            for (std::size_t i = 0; i < sequence.frames.size(); ++i) {
                const auto& frame = sequence.frames[i];
                cv::Mat image = cv::imread(frame.imagePath.string(), cv::IMREAD_GRAYSCALE);
                if (image.empty()) continue;

                // Optical shock test injection (e.g. 0-lux pitch blackout or solar glare)
                if (!testShock.empty() && i >= 20 && i <= 35) {
                    image = adaptive_slam::applyIlluminationShock(image, testShock);
                }

                // UAV Ground Truth Pose for Realistic Multi-Modal LiDAR Raycasting
                cv::Vec3d uavPos(0.0, 0.0, 0.5);
                cv::Matx33d uavRot = cv::Matx33d::eye();
                if (sequence.groundTruth.has_value() && i < sequence.groundTruth->positions.size()) {
                    uavPos = sequence.groundTruth->positions[i];
                }

                // Simulate realistic Livox Avia multi-modal point cloud (250 range points)
                adaptive_slam::LidarScan scan = simulateAviaScan(frame.timestampNs, uavPos, uavRot, 250);

                // High-rate IMU pre-integration step
                adaptive_slam::ImuMeasurement imu;
                imu.timestampNs = frame.timestampNs;
                imu.linearAccel = {0.0, 0.0, 9.81}; // Gravity compensated
                imu.angularVel = {0.0, 0.0, 0.0};
                hybridSlam.processImu(imu);

                // Process synchronized multi-modal frame
                auto state = hybridSlam.processFrame(image, frame.timestampNs, scan);

                if (state.mode == adaptive_slam::HybridPowerMode::LowCost) ++lowCostFrames;
                else if (state.mode == adaptive_slam::HybridPowerMode::Recovery) ++recoveryFrames;
                else if (state.mode == adaptive_slam::HybridPowerMode::VisualBypassLIO) ++bypassFrames;

                // Compute instantaneous error relative to ground truth
                double currentAte = 0.0;
                if (sequence.groundTruth.has_value() && i < sequence.groundTruth->positions.size()) {
                    currentAte = cv::norm(state.positionWorld - sequence.groundTruth->positions[i]);
                }

                // Render FAST-LIVO2 RViz Simulation Frame
                renderFastLivo2Simulation(
                    rvizCanvas,
                    image,
                    hybridSlam.currentTrackedPoints(),
                    hybridSlam.globalLidarMap(),
                    hybridSlam.currentScanWorld(),
                    hybridSlam.trajectory(),
                    state,
                    static_cast<int>(i + 1),
                    static_cast<int>(sequence.frames.size()),
                    currentAte
                );

                if (showLiveGui) {
                    try {
                        cv::imshow(windowName, rvizCanvas);
                        cv::waitKey(1);
                    } catch (...) {}
                }

                if (i == 0 || (i + 1) % printInterval == 0 || i + 1 == sequence.frames.size()) {
                    std::cout << "[RViz] Fixed Frame: map  |  Displays: PointCloud2 /cloud_registered, Odometry /state_estimation/odometry\n"
                              << "[3D] frame=" << (i + 1) << '/' << sequence.frames.size()
                              << " mode=" << adaptive_slam::hybridModeName(state.mode)
                              << " confidence=" << std::fixed << std::setprecision(3) << state.confidence
                              << " visual_inliers=" << state.visualInliers
                              << " lidar_inliers=" << state.lidarInliers
                              << " trajectory=(" << std::setprecision(3) << state.positionWorld[0]
                              << "," << state.positionWorld[1] << "," << state.positionWorld[2] << ")"
                              << " ATE=" << std::setprecision(3) << currentAte << " m"
                              << " MACs=" << state.estimatedMacCount
                              << " time=" << std::setprecision(2) << state.elapsedMs << " ms\n";
                }

                if (replayFps > 0.0 && i + 1 < sequence.frames.size()) {
                    const auto framePeriodMs = static_cast<int>(std::lround(1000.0 / replayFps));
                    std::this_thread::sleep_for(std::chrono::milliseconds(framePeriodMs));
                }
            }

            // Save visualizer snapshot artifact matching FAST-LIVO2 RViz
            const auto snapshotPath = outputPath / "fast_livo2_simulation.jpg";
            cv::imwrite(snapshotPath.string(), rvizCanvas);
            std::cout << "\n[EcoLIVO Simulation] Saved RViz visualizer snapshot to: "
                      << std::filesystem::absolute(snapshotPath).string() << "\n";

            std::cout << "UAV Hybrid SLAM Completed:\n"
                      << "  Total Frames Processed   : " << sequence.frames.size() << '\n'
                      << "  Nominal Low-Cost Frames  : " << lowCostFrames << '\n'
                      << "  Integer CLAHE Recovery   : " << recoveryFrames << '\n'
                      << "  Visual Bypass (0 MACs)   : " << bypassFrames << '\n'
                      << "  Total MAC Operations     : " << hybridSlam.totalMacOperations() << '\n'
                      << "  Total Estimated Energy   : " << std::fixed << std::setprecision(3)
                      << hybridSlam.totalEnergyJoules() << " Joules\n"
                      << "=======================================================\n\n";
        }

        const auto summaries = adaptive_slam::runBenchmark(sequencePath, outputPath, degradations);
        for (const auto& summary : summaries) {
            std::cout << std::setw(10) << summary.configuration << " / "
                      << std::setw(9) << std::left << summary.degradation
                      << std::right << " ATE=";
            if (std::isfinite(summary.ateRmse)) {
                std::cout << std::fixed << std::setprecision(4)
                          << summary.ateRmse << " m";
            } else {
                std::cout << "unavailable";
            }
            std::cout << " latency=" << std::fixed << std::setprecision(2)
                      << summary.meanLatencyMs << " ms"
                      << " failures=" << summary.trackingFailures
                      << " landmarks=" << summary.landmarks << '\n';
        }
        std::cout << "Wrote benchmark artifacts to "
                  << std::filesystem::absolute(outputPath).string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "adaptive_slam: " << error.what() << '\n';
        return 1;
    }
}
