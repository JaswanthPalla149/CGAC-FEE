#include "adaptive_slam/slam.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

namespace {

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
        << " --sequence PATH [--output PATH] [--degradations LIST] [--threads N] [--uav-hybrid] [--test-shock SHOCK] [--fps N]\n"
        << "  LIST: comma-separated none,blur,low_light,noise\n"
        << "  SHOCK: blackout, solar_flare, strobe\n"
        << "  Example: " << executable
        << " --sequence C:\\datasets\\MH_01_easy --output results --uav-hybrid\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path sequencePath;
        std::filesystem::path outputPath{"results"};
        std::vector<std::string> degradations{"none"};
        int threads = 1;
        bool runUavHybrid = false;
        std::string testShock;
        double replayFps = 0.0;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            if (argument == "--uav-hybrid") {
                runUavHybrid = true;
                continue;
            }
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            const std::string value = argv[++i];
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

            int lowCostFrames = 0;
            int recoveryFrames = 0;
            int bypassFrames = 0;

            for (std::size_t i = 0; i < sequence.frames.size(); ++i) {
                const auto& frame = sequence.frames[i];
                cv::Mat image = cv::imread(frame.imagePath.string(), cv::IMREAD_GRAYSCALE);
                if (image.empty()) continue;

                // If test shock is requested, apply on frames 20 to 35
                if (!testShock.empty() && i >= 20 && i <= 35) {
                    image = adaptive_slam::applyIlluminationShock(image, testShock);
                }

                // Simulate lightweight 3D LiDAR point cloud (150 metric range points)
                adaptive_slam::LidarScan scan;
                scan.timestampNs = frame.timestampNs;
                for (int p = 0; p < 150; ++p) {
                    scan.points.push_back({
                        static_cast<float>(std::cos(p * 0.05) * 5.0),
                        static_cast<float>(std::sin(p * 0.05) * 5.0),
                        static_cast<float>(-0.5 + (p % 10) * 0.1),
                        100.0f
                    });
                }

                auto state = hybridSlam.processFrame(image, frame.timestampNs, scan);
                if (state.mode == adaptive_slam::HybridPowerMode::LowCost) ++lowCostFrames;
                else if (state.mode == adaptive_slam::HybridPowerMode::Recovery) ++recoveryFrames;
                else if (state.mode == adaptive_slam::HybridPowerMode::VisualBypassLIO) ++bypassFrames;
                if (replayFps > 0.0 && i + 1 < sequence.frames.size()) {
                    const auto framePeriodMs = static_cast<int>(
                        std::lround(1000.0 / replayFps));
                    std::this_thread::sleep_for(std::chrono::milliseconds(framePeriodMs));
                }
            }

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

        const auto summaries =
            adaptive_slam::runBenchmark(sequencePath, outputPath, degradations);
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
