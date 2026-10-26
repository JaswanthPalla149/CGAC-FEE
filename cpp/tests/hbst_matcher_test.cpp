#include "adaptive_slam/hbst_matcher.hpp"
#include "adaptive_slam/low_mac_features.hpp"

#include <iostream>
#include <cassert>
#include <random>

int main() {
    using namespace adaptive_slam;

    // 1. Test BinaryDescriptor Hamming Distance (Zero-MAC)
    BinaryDescriptor d1, d2;
    d1.words[0] = 0x0000000000000000ULL;
    d2.words[0] = 0x000000000000000FULL; // 4 bits different
    assert(d1.hammingDistance(d2) == 4);

    d1.setBit(100, true);
    assert(d1.getBit(100) == true);
    assert(d2.getBit(100) == false);
    assert(d1.hammingDistance(d2) == 5);

    // 2. Test HBST Tree Construction and Query
    std::vector<BinaryDescriptor> dataset(32);
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        dataset[i].words[0] = static_cast<std::uint64_t>(i);
        dataset[i].words[1] = static_cast<std::uint64_t>(i * 13);
        dataset[i].words[2] = 0;
        dataset[i].words[3] = 0;
    }

    HBSTTree tree;
    tree.build(dataset);
    assert(tree.featureCount() == 32);

    // Query exact match for item 17
    BinaryDescriptor query = dataset[17];
    MatchResult result = tree.query(query);
    assert(result.feature_idx == 17);
    assert(result.distance == 0);

    // Query item with 1 bit difference
    query.setBit(200, !query.getBit(200));
    result = tree.query(query);
    assert(result.feature_idx == 17);
    assert(result.distance == 1);

    // 3. Test Integer-Only CLAHE
    cv::Mat darkImage(120, 160, CV_8UC1, cv::Scalar(25)); // Low light image
    // Add a brighter gradient patch
    for (int y = 40; y < 80; ++y) {
        for (int x = 50; x < 110; ++x) {
            darkImage.at<std::uint8_t>(y, x) = 60;
        }
    }
    cv::Mat enhanced = applyIntegerClahe(darkImage, 3, 4, 4);
    assert(enhanced.size() == darkImage.size());
    assert(enhanced.type() == CV_8UC1);
    // Contrast should be expanded
    double minVal, maxVal;
    cv::minMaxLoc(enhanced, &minVal, &maxVal);
    assert(maxVal > 60.0); // Contrast successfully equalized

    // 4. Test Low-MAC BRIEF Descriptors
    std::vector<cv::Point2f> kpts{{60.0f, 60.0f}, {80.0f, 50.0f}};
    auto descriptors = extractBinaryDescriptors(enhanced, kpts);
    assert(descriptors.size() == 2);

    std::cout << "All HBST & Low-MAC feature tests PASSED successfully!\n";
    return 0;
}
