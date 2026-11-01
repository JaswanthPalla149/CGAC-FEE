#ifndef HBST_MATCHER_H
#define HBST_MATCHER_H

#include <ap_int.h>
#include <hls_stream.h>

// Configuration Constants for 256-bit ORB Descriptors
#define DESCRIPTOR_BITS 256
#define DESCRIPTOR_WORDS (DESCRIPTOR_BITS / 32) // 8 uint32_t words per ORB descriptor
#define MAX_FEATURES 1024
#define MAX_TREE_NODES 2048

// Heterogeneous Node Bit Masks for HBTA Encoding
#define LEAF_FLAG_BIT 31
#define INDEX_MASK 0x7FFFFFFF

// Structs for Hardware Interface
typedef ap_uint<256> orb_descriptor_t;

struct MatchResult {
    int feature_idx;
    uint16_t distance;
};

// Top-Level Hardware Function Prototype
void hbst_tree_search(
    const uint32_t tree_mem[MAX_TREE_NODES],
    const uint32_t leaf_info_mem[MAX_TREE_NODES][2], // [0]: start_idx, [1]: length
    const orb_descriptor_t feature_mem[MAX_FEATURES],
    orb_descriptor_t query,
    MatchResult *result
);

#endif // HBST_MATCHER_H