#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

namespace rtp_llm {

// Cache-internal physical entry geometry; fixed-state specs own this value.
struct ResolvedStateEntryLayout {
    uint32_t seq_size_per_block = 0;  // original tokens per physical STATE block
    uint32_t full_entries       = 0;  // aligned entries in the complete ring
    uint32_t local_entry_begin  = 0;  // first complete-ring entry on this prefill rank
    uint32_t local_entries      = 0;  // physical entries in this process's block
    size_t   entry_bytes        = 0;
};

// Physical packed-byte projection for BLOCK_STRIDE fixed regions. Unlike a
// STATE entry slice, a rank boundary may fall inside an FP8 data/scale row.
struct ResolvedSwaByteLayout {
    uint32_t seq_size_per_block = 0;  // original tokens per physical SWA block
    uint32_t full_entries       = 0;  // complete logical ring, before byte slicing
    size_t   full_payload_bytes = 0;  // data and scale, excluding trailing padding
    size_t   full_stride_bytes  = 0;  // data, scale and trailing padding
    size_t   local_byte_begin   = 0;  // offset in the complete packed block
    size_t   local_stride_bytes = 0;  // rank-local allocated bytes
};

// Physical span and per-kernel-page entries after cache CP resolution.
struct ResolvedCompressedLayout {
    uint32_t seq_size_per_block       = 0;  // original tokens per physical OpaqueKV block
    uint32_t entries_per_kernel_block = 0;  // complete entries in each kernel page before physical aggregation
};

struct NoSpecProjection {};
using SpecPhysicalLayout = std::variant<std::monostate,
                                        NoSpecProjection,
                                        ResolvedStateEntryLayout,
                                        ResolvedSwaByteLayout,
                                        ResolvedCompressedLayout>;

bool sameSpecPhysicalLayout(const SpecPhysicalLayout& lhs, const SpecPhysicalLayout& rhs);

}  // namespace rtp_llm
