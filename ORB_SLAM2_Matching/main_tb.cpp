#include <iostream>
#include <cstdlib>
#include "hbst_matcher.h"

int main() {
    uint32_t tree_mem[MAX_TREE_NODES] = {0};
    uint32_t leaf_info_mem[MAX_TREE_NODES][2] = {{0}};
    orb_descriptor_t feature_mem[MAX_FEATURES];

    // 1. Populate Feature Memory with mock 256-bit binary descriptors
    for (int i = 0; i < 16; i++) {
        orb_descriptor_t feat = 0;
        if (i < 8) {
            feat = 0x00000000; // Left partition (bit 0 is 0)
            feat[10] = (i & 1); 
        } else {
            feat = 0xFFFFFFFF; // Right partition (bit 0 is 1)
            feat[0] = 1;       // Ensure bit 0 is set
        }
        feature_mem[i] = feat;
    }

    // 2. Build a Mock HBST Tree in Eytzinger Array Layout
    // Root Node (index 0): Tests bit index 0
    tree_mem[0] = 0 & INDEX_MASK; // Internal node (bit 31 = 0), check index 0

    // Left Child (index 1): Leaf for features 0 to 7
    tree_mem[1] = (1U << LEAF_FLAG_BIT); // Leaf node (bit 31 = 1)
    leaf_info_mem[1][0] = 0; // Start at feature 0
    leaf_info_mem[1][1] = 8; // Length 8

    // Right Child (index 2): Leaf for features 8 to 15
    tree_mem[2] = (1U << LEAF_FLAG_BIT); // Leaf node (bit 31 = 1)
    leaf_info_mem[2][0] = 8; // Start at feature 8
    leaf_info_mem[2][1] = 8; // Length 8

    // 3. Define Query (Matches features in the right partition)
    orb_descriptor_t query = 0xFFFFFFFF;
    query[0] = 1; // Bit 0 = 1 -> Should traverse to Node 2 (Right Child)

    MatchResult result;

    // 4. Run HLS Hardware Function
    std::cout << "--> Running HBST Matcher C-Simulation..." << std::endl;
    hbst_tree_search(tree_mem, leaf_info_mem, feature_mem, query, &result);

    // 5. Verify Results
    std::cout << "Matched Feature Index: " << result.feature_idx << std::endl;
    std::cout << "Hamming Distance     : " << result.distance << std::endl;

    if (result.feature_idx >= 8 && result.feature_idx <= 15) {
        std::cout << "TEST PASSED: Correctly traversed tree and matched leaf node!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED: Incorrect match!" << std::endl;
        return 1;
    }
}