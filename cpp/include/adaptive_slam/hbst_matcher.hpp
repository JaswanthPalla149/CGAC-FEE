#ifndef ADAPTIVE_SLAM_HBST_MATCHER_HPP
#define ADAPTIVE_SLAM_HBST_MATCHER_HPP

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace adaptive_slam {

// Configuration constants matching ORB_SLAM2_Matching/hbst_matcher.h
constexpr int kDescriptorBits = 256;
constexpr int kDescriptorWords = kDescriptorBits / 64; // 4 uint64_t words
constexpr int kMaxFeatures = 1024;
constexpr int kMaxTreeNodes = 2048;

constexpr uint32_t kLeafFlagBit = 31;
constexpr uint32_t kIndexMask = 0x7FFFFFFF;

inline int popcount64(std::uint64_t v) noexcept {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(v));
#else
    return __builtin_popcountll(v);
#endif
}

// 256-bit Binary Descriptor (Hardware-acceleratable via bitwise XOR + PopCount)
struct alignas(32) BinaryDescriptor {
    std::uint64_t words[4]{0, 0, 0, 0};

    bool getBit(int bitIndex) const noexcept {
        if (bitIndex < 0 || bitIndex >= kDescriptorBits) return false;
        const int wordIdx = bitIndex / 64;
        const int bitInWord = bitIndex % 64;
        return (words[wordIdx] >> bitInWord) & 1ULL;
    }

    void setBit(int bitIndex, bool val) noexcept {
        if (bitIndex < 0 || bitIndex >= kDescriptorBits) return;
        const int wordIdx = bitIndex / 64;
        const int bitInWord = bitIndex % 64;
        if (val) {
            words[wordIdx] |= (1ULL << bitInWord);
        } else {
            words[wordIdx] &= ~(1ULL << bitInWord);
        }
    }

    // Zero-MAC Hamming distance: 4 XORs + 4 Popcounts = 0 float operations!
    inline std::uint16_t hammingDistance(const BinaryDescriptor& other) const noexcept {
        return static_cast<std::uint16_t>(
            popcount64(words[0] ^ other.words[0]) +
            popcount64(words[1] ^ other.words[1]) +
            popcount64(words[2] ^ other.words[2]) +
            popcount64(words[3] ^ other.words[3])
        );
    }
};

struct MatchResult {
    int feature_idx{-1};
    std::uint16_t distance{257}; // Max possible distance + 1
};

// Top-level hardware function implementation (compatible with HLS HBST architecture)
inline void hbst_tree_search(
    const std::uint32_t tree_mem[kMaxTreeNodes],
    const std::uint32_t leaf_info_mem[kMaxTreeNodes][2],
    const BinaryDescriptor feature_mem[kMaxFeatures],
    const BinaryDescriptor& query,
    MatchResult* result
) {
    if (!result) return;

    std::uint32_t curr_node = 0; // Eytzinger array root
    bool is_leaf = false;
    std::uint32_t bit_to_check = 0;

    // Stage 1: Logarithmic Tree Traversal
    for (int depth = 0; depth < 16; ++depth) {
        if (curr_node >= kMaxTreeNodes) break;
        const std::uint32_t node_data = tree_mem[curr_node];
        is_leaf = (node_data >> kLeafFlagBit) & 0x1;
        if (is_leaf) {
            break;
        }

        bit_to_check = node_data & kIndexMask;
        const bool query_bit = query.getBit(static_cast<int>(bit_to_check));

        // Eytzinger navigation: Left = 2*i + 1, Right = 2*i + 2
        if (!query_bit) {
            curr_node = (curr_node << 1) + 1;
        } else {
            curr_node = (curr_node << 1) + 2;
        }
    }

    if (curr_node >= kMaxTreeNodes) {
        result->feature_idx = -1;
        result->distance = 257;
        return;
    }

    // Read leaf region parameters
    const std::uint32_t start_idx = leaf_info_mem[curr_node][0];
    const std::uint32_t length    = leaf_info_mem[curr_node][1];

    // Stage 2: Hardware leaf candidate comparison
    std::uint16_t min_dist = 257;
    int best_match_idx = -1;

    for (std::uint32_t i = 0; i < length && (start_idx + i) < kMaxFeatures; ++i) {
        const std::uint32_t feat_addr = start_idx + i;
        const std::uint16_t dist = query.hammingDistance(feature_mem[feat_addr]);
        if (dist < min_dist) {
            min_dist = dist;
            best_match_idx = static_cast<int>(feat_addr);
        }
    }

    result->feature_idx = best_match_idx;
    result->distance = min_dist;
}

// C++ Index Builder that generates the hardware Eytzinger tables
class HBSTTree {
public:
    HBSTTree() {
        std::memset(tree_mem_, 0, sizeof(tree_mem_));
        std::memset(leaf_info_mem_, 0, sizeof(leaf_info_mem_));
        std::memset(feature_mem_, 0, sizeof(feature_mem_));
    }

    void build(const std::vector<BinaryDescriptor>& features) {
        feature_count_ = std::min(features.size(), static_cast<std::size_t>(kMaxFeatures));
        for (std::size_t i = 0; i < feature_count_; ++i) {
            feature_mem_[i] = features[i];
        }

        std::vector<std::uint32_t> indices(feature_count_);
        for (std::size_t i = 0; i < feature_count_; ++i) indices[i] = static_cast<std::uint32_t>(i);

        std::memset(tree_mem_, 0, sizeof(tree_mem_));
        std::memset(leaf_info_mem_, 0, sizeof(leaf_info_mem_));

        if (feature_count_ == 0) return;

        buildNode(0, indices, 0);
    }

    MatchResult query(const BinaryDescriptor& descriptor) const {
        MatchResult res;
        hbst_tree_search(tree_mem_, leaf_info_mem_, feature_mem_, descriptor, &res);
        return res;
    }

    const std::uint32_t* treeMem() const noexcept { return tree_mem_; }
    const std::uint32_t (*leafInfoMem() const noexcept)[2] { return leaf_info_mem_; }
    const BinaryDescriptor* featureMem() const noexcept { return feature_mem_; }
    std::size_t featureCount() const noexcept { return feature_count_; }

private:
    void buildNode(std::uint32_t nodeIdx, const std::vector<std::uint32_t>& indices, int depth) {
        if (nodeIdx >= kMaxTreeNodes) return;

        // Leaf threshold: <= 16 features or max depth reached
        if (indices.size() <= 16 || depth >= 14) {
            tree_mem_[nodeIdx] = (1U << kLeafFlagBit);
            if (!indices.empty()) {
                leaf_info_mem_[nodeIdx][0] = indices.front();
                leaf_info_mem_[nodeIdx][1] = static_cast<std::uint32_t>(indices.size());
            } else {
                leaf_info_mem_[nodeIdx][0] = 0;
                leaf_info_mem_[nodeIdx][1] = 0;
            }
            return;
        }

        // Find split bit that maximizes entropy (closest to 50% ones)
        int bestBit = 0;
        int bestDiff = static_cast<int>(indices.size());
        for (int b = 0; b < kDescriptorBits; ++b) {
            int ones = 0;
            for (std::uint32_t idx : indices) {
                if (feature_mem_[idx].getBit(b)) ++ones;
            }
            const int zeros = static_cast<int>(indices.size()) - ones;
            const int diff = std::abs(ones - zeros);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestBit = b;
            }
        }

        // Partition
        std::vector<std::uint32_t> left;
        std::vector<std::uint32_t> right;
        for (std::uint32_t idx : indices) {
            if (!feature_mem_[idx].getBit(bestBit)) {
                left.push_back(idx);
            } else {
                right.push_back(idx);
            }
        }

        if (left.empty() || right.empty()) {
            tree_mem_[nodeIdx] = (1U << kLeafFlagBit);
            leaf_info_mem_[nodeIdx][0] = indices.front();
            leaf_info_mem_[nodeIdx][1] = static_cast<std::uint32_t>(indices.size());
            return;
        }

        tree_mem_[nodeIdx] = static_cast<std::uint32_t>(bestBit) & kIndexMask;
        buildNode((nodeIdx << 1) + 1, left, depth + 1);
        buildNode((nodeIdx << 1) + 2, right, depth + 1);
    }

    std::uint32_t tree_mem_[kMaxTreeNodes];
    std::uint32_t leaf_info_mem_[kMaxTreeNodes][2];
    BinaryDescriptor feature_mem_[kMaxFeatures];
    std::size_t feature_count_{0};
};

} // namespace adaptive_slam

#endif // ADAPTIVE_SLAM_HBST_MATCHER_HPP
