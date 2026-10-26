#ifndef ADAPTIVE_SLAM_LOW_MAC_FEATURES_HPP
#define ADAPTIVE_SLAM_LOW_MAC_FEATURES_HPP

#include "adaptive_slam/hbst_matcher.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace adaptive_slam {

// Fixed-Point arithmetic helpers (Q8.8)
using fixed16_t = std::int16_t;
constexpr int kFixedShift = 8;
constexpr fixed16_t kFixedOne = 1 << kFixedShift;

// Predefined 256-pair BRIEF test pattern for 31x31 patch (Deterministic, hardware-friendly)
struct BriefPattern {
    static constexpr int kNumPairs = 256;
    // Offsets within [-15, 15] for patch center
    std::int8_t p1_x[kNumPairs];
    std::int8_t p1_y[kNumPairs];
    std::int8_t p2_x[kNumPairs];
    std::int8_t p2_y[kNumPairs];

    static const BriefPattern& instance() {
        static BriefPattern pattern = []() {
            BriefPattern p{};
            // Deterministic pseudo-random generation of Gaussian sampling pairs
            std::uint32_t state = 0x12345678;
            auto lcg = [&state]() -> int {
                state = state * 1664525U + 1013904223U;
                int val = static_cast<int>((state >> 16) & 0x7FFF);
                return (val % 29) - 14; // in [-14, 14]
            };
            for (int i = 0; i < kNumPairs; ++i) {
                p.p1_x[i] = static_cast<std::int8_t>(lcg());
                p.p1_y[i] = static_cast<std::int8_t>(lcg());
                p.p2_x[i] = static_cast<std::int8_t>(lcg());
                p.p2_y[i] = static_cast<std::int8_t>(lcg());
            }
            return p;
        }();
        return pattern;
    }
};

// 1. Integer-Only CLAHE (Contrast-Limited Adaptive Histogram Equalization)
// Pure integer operations: 0 floating point MACs
inline cv::Mat applyIntegerClahe(const cv::Mat& gray, int clipLimit = 3, int gridTilesX = 8, int gridTilesY = 8) {
    if (gray.empty() || gray.type() != CV_8UC1) {
        return gray.clone();
    }

    const int width = gray.cols;
    const int height = gray.rows;
    cv::Mat output(height, width, CV_8UC1);

    const int tileW = width / gridTilesX;
    const int tileH = height / gridTilesY;
    if (tileW <= 0 || tileH <= 0) return gray.clone();

    // Allocate tile CDFs: gridTilesY x gridTilesX x 256
    std::vector<std::array<std::uint8_t, 256>> tileCdfs(gridTilesX * gridTilesY);
    const int tileSize = tileW * tileH;
    const int clipCount = std::max(1, (clipLimit * tileSize) / 256);

    for (int ty = 0; ty < gridTilesY; ++ty) {
        for (int tx = 0; tx < gridTilesX; ++tx) {
            const int startX = tx * tileW;
            const int startY = ty * tileH;
            const int endX = (tx == gridTilesX - 1) ? width : startX + tileW;
            const int endY = (ty == gridTilesY - 1) ? height : startY + tileH;
            const int actualTileSize = (endX - startX) * (endY - startY);

            // Step 1: Histogram calculation (Integer only)
            std::uint32_t hist[256]{0};
            for (int y = startY; y < endY; ++y) {
                const std::uint8_t* row = gray.ptr<std::uint8_t>(y);
                for (int x = startX; x < endX; ++x) {
                    ++hist[row[x]];
                }
            }

            // Step 2: Integer clipping and excess accumulation
            std::uint32_t excess = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > static_cast<std::uint32_t>(clipCount)) {
                    excess += (hist[i] - static_cast<std::uint32_t>(clipCount));
                    hist[i] = static_cast<std::uint32_t>(clipCount);
                }
            }

            // Step 3: Redistribute excess
            const std::uint32_t bonus = excess >> 8; // Integer divide by 256
            for (int i = 0; i < 256; ++i) {
                hist[i] += bonus;
            }

            // Step 4: Build Cumulative Distribution Function (CDF) scaled to [0, 255]
            std::uint32_t cdf = 0;
            auto& lut = tileCdfs[ty * gridTilesX + tx];
            for (int i = 0; i < 256; ++i) {
                cdf += hist[i];
                std::uint32_t val = (cdf * 255U) / actualTileSize;
                lut[i] = static_cast<std::uint8_t>(std::min(255U, val));
            }
        }
    }

    // Step 5: Bilinear fixed-point interpolation across neighboring tiles
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* inRow = gray.ptr<std::uint8_t>(y);
        std::uint8_t* outRow = output.ptr<std::uint8_t>(y);

        // Sub-tile continuous position
        const int fy = y - tileH / 2;
        int ty1 = fy / tileH;
        int ty2 = ty1 + 1;
        const int yFract = ((fy >= 0 ? fy : 0) % tileH) * kFixedOne / tileH;
        ty1 = std::clamp(ty1, 0, gridTilesY - 1);
        ty2 = std::clamp(ty2, 0, gridTilesY - 1);

        for (int x = 0; x < width; ++x) {
            const int fx = x - tileW / 2;
            int tx1 = fx / tileW;
            int tx2 = tx1 + 1;
            const int xFract = ((fx >= 0 ? fx : 0) % tileW) * kFixedOne / tileW;
            tx1 = std::clamp(tx1, 0, gridTilesX - 1);
            tx2 = std::clamp(tx2, 0, gridTilesX - 1);

            const std::uint8_t pixel = inRow[x];

            const int valTopLeft     = tileCdfs[ty1 * gridTilesX + tx1][pixel];
            const int valTopRight    = tileCdfs[ty1 * gridTilesX + tx2][pixel];
            const int valBottomLeft  = tileCdfs[ty2 * gridTilesX + tx1][pixel];
            const int valBottomRight = tileCdfs[ty2 * gridTilesX + tx2][pixel];

            // Bilinear interpolation using integer shifts (0 floating point ops)
            const int top = valTopLeft + (((valTopRight - valTopLeft) * xFract) >> kFixedShift);
            const int bottom = valBottomLeft + (((valBottomRight - valBottomLeft) * xFract) >> kFixedShift);
            const int result = top + (((bottom - top) * yFract) >> kFixedShift);

            outRow[x] = static_cast<std::uint8_t>(std::clamp(result, 0, 255));
        }
    }

    return output;
}

// 2. Low-MAC Binary Descriptor Extraction
// Computes 256-bit binary descriptors from keypoints using integer intensity tests
inline std::vector<BinaryDescriptor> extractBinaryDescriptors(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& keypoints
) {
    const auto& pattern = BriefPattern::instance();
    std::vector<BinaryDescriptor> descriptors(keypoints.size());
    const int width = image.cols;
    const int height = image.rows;

    for (std::size_t k = 0; k < keypoints.size(); ++k) {
        const int cx = static_cast<int>(std::round(keypoints[k].x));
        const int cy = static_cast<int>(std::round(keypoints[k].y));
        BinaryDescriptor desc;

        // Ensure patch lies inside image
        if (cx < 16 || cx >= width - 16 || cy < 16 || cy >= height - 16) {
            descriptors[k] = desc;
            continue;
        }

        for (int i = 0; i < BriefPattern::kNumPairs; ++i) {
            const int x1 = cx + pattern.p1_x[i];
            const int y1 = cy + pattern.p1_y[i];
            const int x2 = cx + pattern.p2_x[i];
            const int y2 = cy + pattern.p2_y[i];

            const std::uint8_t val1 = image.at<std::uint8_t>(y1, x1);
            const std::uint8_t val2 = image.at<std::uint8_t>(y2, x2);

            // Bit test: 0 Multiplications / 0 MACs
            if (val1 < val2) {
                desc.setBit(i, true);
            }
        }
        descriptors[k] = desc;
    }
    return descriptors;
}

} // namespace adaptive_slam

#endif // ADAPTIVE_SLAM_LOW_MAC_FEATURES_HPP
