#include "hbst_matcher.h"

// Helper function: Compute Hamming Distance using ap_uint native XOR + Popcount
static uint16_t compute_hamming_distance(orb_descriptor_t a, orb_descriptor_t b) {
    #pragma HLS INLINE
    orb_descriptor_t diff = a ^ b;
    
    // Count set bits using hardware bit reduction
    uint16_t dist = 0;
    for (int i = 0; i < DESCRIPTOR_BITS; i++) {
        #pragma HLS UNROLL
        dist += diff[i];
    }
    return dist;
}

void hbst_tree_search(
    const uint32_t tree_mem[MAX_TREE_NODES],
    const uint32_t leaf_info_mem[MAX_TREE_NODES][2],
    const orb_descriptor_t feature_mem[MAX_FEATURES],
    orb_descriptor_t query,
    MatchResult *result
) {
    // Hardware Bus Interface Directives (AXI4-Lite & Memory Interfaces)
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE m_axi port=tree_mem offset=slave bundle=gmem0 depth=2048
    #pragma HLS INTERFACE m_axi port=leaf_info_mem offset=slave bundle=gmem1 depth=4096
    #pragma HLS INTERFACE m_axi port=feature_mem offset=slave bundle=gmem2 depth=1024
    #pragma HLS INTERFACE s_axilite port=query bundle=control
    #pragma HLS INTERFACE s_axilite port=result bundle=control

    uint32_t curr_node = 0; // Start at root node in Eytzinger array layout
    bool is_leaf = false;
    uint32_t bit_to_check = 0;

    // -------------------------------------------------------------
    // STAGE 1: Logarithmic Tree Traversal (Algorithm 1 / Tree Matcher)
    // -------------------------------------------------------------
    TRAVERSE_TREE: for (int depth = 0; depth < 16; depth++) {
        #pragma HLS PIPELINE II=1
        
        uint32_t node_data = tree_mem[curr_node];
        is_leaf = (node_data >> LEAF_FLAG_BIT) & 0x1;
        
        if (is_leaf) {
            break; // Reached leaf subtree root
        }

        bit_to_check = node_data & INDEX_MASK;
        
        // Inspect query descriptor bit at target bit index
        uint8_t query_bit = query[bit_to_check];

        // Eytzinger Layout Navigation: Left = 2*i + 1, Right = 2*i + 2
        if (query_bit == 0) {
            curr_node = (curr_node << 1) + 1;
        } else {
            curr_node = (curr_node << 1) + 2;
        }
    }

    // Read leaf region parameters (Contiguous feature range address & size)
    uint32_t start_idx = leaf_info_mem[curr_node][0];
    uint32_t length    = leaf_info_mem[curr_node][1];

    // -------------------------------------------------------------
    // STAGE 2: BruteForce Matcher across Leaf Node's Features
    // -------------------------------------------------------------
    uint16_t min_dist = 257; // Max possible ORB distance is 256
    int best_match_idx = -1;

    BRUTE_FORCE_LEAF: for (uint32_t i = 0; i < MAX_FEATURES; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=16
        
        if (i >= length) break;

        uint32_t feat_addr = start_idx + i;
        orb_descriptor_t candidate = feature_mem[feat_addr];

        uint16_t dist = compute_hamming_distance(query, candidate);

        if (dist < min_dist) {
            min_dist = dist;
            best_match_idx = feat_addr;
        }
    }

    // Write back hardware results
    result->feature_idx = best_match_idx;
    result->distance = min_dist;
}