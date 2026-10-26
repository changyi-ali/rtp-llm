#include "rtp_llm/cpp/cache/CacheSpecLayoutResolver.h"

#include <limits>
#include <numeric>
#include <stdexcept>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {
enum class Projection {
    NONE,
    STATE,
    SWA,
    COMPRESSED
};

Projection classifySpecProjection(const KVCacheSpecDesc& desc) {
    if ((desc.cache_type != KVCacheSpecType::OpaqueState && desc.cache_type != KVCacheSpecType::OpaqueKV) || !desc.cp) {
        return Projection::NONE;
    }
    const auto slice = desc.cp->prefill_slice_layout.value_or(CpPrefillSliceLayout::NONE);
    RTP_LLM_CHECK_WITH_INFO(slice == CpPrefillSliceLayout::NONE || slice == CpPrefillSliceLayout::PAYLOAD
                                || slice == CpPrefillSliceLayout::BLOCK_STRIDE,
                            "cache tag=%s has invalid prefill slice layout",
                            desc.tag.c_str());
    if (!desc.cp->scale_seq_size.value_or(false) && !desc.cp->align_payload.value_or(false)
        && slice == CpPrefillSliceLayout::NONE) {
        return Projection::NONE;
    }
    if (desc.cache_type == KVCacheSpecType::OpaqueKV) {
        return Projection::COMPRESSED;
    }
    return slice == CpPrefillSliceLayout::BLOCK_STRIDE ? Projection::SWA : Projection::STATE;
}

uint32_t fixedRegionCpSize(const ParallelismConfig& parallelism) {
    if (!parallelism.prefill_cp_config.kv_cache_sharded) {
        return 1;
    }
    int64_t cp = 1;
    if (parallelism.role_type == RoleType::PREFILL && parallelism.tp_size > 1) {
        cp = parallelism.tp_size;
    } else if (parallelism.role_type == RoleType::DECODE && parallelism.prefill_cp_config.is_prefill_enabled()) {
        cp = parallelism.prefill_cp_config.prefill_cp_size;
        if (cp <= 1) {
            throw std::invalid_argument("fixed region decode CP layout requires explicit prefill_cp_size");
        }
    }
    if (cp > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("fixed region CP size exceeds uint32 range");
    }
    return static_cast<uint32_t>(cp);
}

uint64_t ringEntries(const KVCacheSpecDesc& desc, uint32_t gen_num_per_cycle) {
    const uint64_t cycles = uint64_t{1} + desc.state_ring_overlap;
    const uint64_t extra  = desc.state_ring_include_gen_num_per_cycle ? gen_num_per_cycle : 0;
    const auto     max    = std::numeric_limits<uint64_t>::max();
    if (desc.compression_ratio != 0 && cycles > (max - extra - 1) / desc.compression_ratio) {
        throw std::invalid_argument("fixed region ring entry count overflows uint64");
    }
    return (cycles * desc.compression_ratio + extra + 1) & ~uint64_t{1};
}

uint64_t alignEntries(uint64_t entries, uint32_t cp_size) {
    if (entries > std::numeric_limits<uint64_t>::max() - (cp_size - 1)) {
        throw std::invalid_argument("fixed region aligned entry count overflows uint64");
    }
    return (entries + cp_size - 1) / cp_size * cp_size;
}

size_t checkedEntryBytes(const KVCacheSpecDesc& desc, const char* error) {
    const auto width = getTypeSize(desc.entry_dtype);
    if (width == 0 || desc.entry_elems == 0 || desc.entry_elems > std::numeric_limits<size_t>::max() / width) {
        throw std::invalid_argument(error);
    }
    return static_cast<size_t>(desc.entry_elems) * width;
}

size_t byteAlignment(size_t alignment, uint32_t cp_size) {
    const auto quotient = alignment / std::gcd(alignment, static_cast<size_t>(cp_size));
    if (quotient > std::numeric_limits<size_t>::max() / cp_size) {
        throw std::invalid_argument("SWA packed-byte stride overflows size_t");
    }
    return quotient * cp_size;
}

ResolvedStateEntryLayout
resolveState(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx, const ParallelismConfig& parallelism) {
    const auto base_block_tokens   = ctx.seq_size_per_block;
    const auto kernel_block_tokens = ctx.kernel_tokens_per_block;
    const auto gen_num_per_cycle   = ctx.gen_num_per_cycle;
    if (base_block_tokens == 0) {
        throw std::invalid_argument("STATE layout requires positive base block tokens");
    }
    const auto entry_bytes = checkedEntryBytes(desc, "fixed entry projection requires a valid entry format");
    if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING && desc.compression_ratio == 0) {
        throw std::invalid_argument("fixed entry projection requires a valid entry format");
    }
    const uint32_t cp_size = fixedRegionCpSize(parallelism);
    uint64_t       raw     = desc.explicit_entry_count;
    if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
        raw = ringEntries(desc, gen_num_per_cycle);
    } else if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
        if (desc.compression_ratio == 0 || kernel_block_tokens == 0
            || kernel_block_tokens % desc.compression_ratio != 0) {
            throw std::invalid_argument("fixed entry CP projection requires K divisible by ratio");
        }
        raw = kernel_block_tokens / desc.compression_ratio;
    }
    if (raw == 0) {
        throw std::invalid_argument("fixed entry projection requires positive entries");
    }
    const bool align =
        desc.cp->align_payload.value_or(false) || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD;
    const uint64_t aligned       = align ? alignEntries(raw, cp_size) : raw;
    const bool     prefill_slice = desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD && cp_size > 1
                               && parallelism.role_type == RoleType::PREFILL;
    const uint64_t local = prefill_slice ? aligned / cp_size : aligned;
    const uint64_t seq =
        static_cast<uint64_t>(base_block_tokens) * (desc.cp->scale_seq_size.value_or(false) ? cp_size : 1);
    if (aligned > std::numeric_limits<uint32_t>::max() || local > std::numeric_limits<uint32_t>::max()
        || seq > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("STATE entry projection exceeds uint32 range");
    }
    const int64_t rank = prefill_slice ? parallelism.tp_rank : 0;
    if (rank < 0 || static_cast<uint64_t>(rank) >= cp_size) {
        throw std::invalid_argument("STATE entry projection rank exceeds CP size");
    }
    ResolvedStateEntryLayout value{static_cast<uint32_t>(seq),
                                   static_cast<uint32_t>(aligned),
                                   static_cast<uint32_t>(rank * local),
                                   static_cast<uint32_t>(local),
                                   entry_bytes};
    return value;
}

ResolvedCompressedLayout
resolveCompressed(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx, const ParallelismConfig& parallelism) {
    const auto base_block_tokens   = ctx.seq_size_per_block;
    const auto kernel_block_tokens = ctx.kernel_tokens_per_block;
    const auto gen_num_per_cycle   = ctx.gen_num_per_cycle;
    if (base_block_tokens == 0) {
        throw std::invalid_argument("compressed layout requires positive base block tokens");
    }
    const uint32_t cp_size = fixedRegionCpSize(parallelism);
    const uint64_t seq =
        static_cast<uint64_t>(base_block_tokens) * (desc.cp->scale_seq_size.value_or(false) ? cp_size : 1);
    if (seq > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("compressed CP block token span exceeds uint32 range");
    }
    const uint32_t kernel  = SpecBuilder::groupType(desc) != CacheGroupType::FULL || kernel_block_tokens == 0 ?
                                 static_cast<uint32_t>(seq) :
                                 std::min(kernel_block_tokens, static_cast<uint32_t>(seq));
    uint64_t       entries = desc.explicit_entry_count;
    if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
        if (desc.compression_ratio == 0 || kernel == 0 || kernel % desc.compression_ratio != 0) {
            throw std::invalid_argument("compressed CP projection requires K divisible by ratio");
        }
        entries = kernel / desc.compression_ratio;
    } else if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
        if (desc.compression_ratio == 0) {
            throw std::invalid_argument("compressed CP projection requires positive ratio");
        }
        entries = ringEntries(desc, gen_num_per_cycle);
    }
    const bool align =
        cp_size > 1
        && (desc.cp->align_payload.value_or(false) || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD);
    if (align) {
        entries = alignEntries(entries, cp_size);
        if (desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD
            && parallelism.role_type == RoleType::PREFILL) {
            entries /= cp_size;
        }
    }
    if (entries == 0 || entries > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("compressed CP entry count exceeds uint32 range");
    }
    const ResolvedCompressedLayout value{static_cast<uint32_t>(seq), static_cast<uint32_t>(entries)};
    return value;
}

ResolvedSwaByteLayout
resolveSwa(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx, const ParallelismConfig& parallelism) {
    const auto base_block_tokens   = ctx.seq_size_per_block;
    const auto kernel_block_tokens = ctx.kernel_tokens_per_block;
    const auto gen_num_per_cycle   = ctx.gen_num_per_cycle;
    if (base_block_tokens == 0) {
        throw std::invalid_argument("SWA layout requires positive base block tokens");
    }
    const auto entry_bytes = checkedEntryBytes(desc, "SWA packed-byte projection requires a valid entry format");
    if (desc.entry_count_mode != OpaqueBlockEntryCountMode::EXPLICIT && desc.compression_ratio == 0) {
        throw std::invalid_argument("SWA packed-byte projection requires a valid entry format");
    }
    const uint32_t cp_size     = fixedRegionCpSize(parallelism);
    uint64_t       raw_entries = desc.explicit_entry_count;
    if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
        raw_entries = ringEntries(desc, gen_num_per_cycle);
    } else if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
        if (kernel_block_tokens == 0 || kernel_block_tokens % desc.compression_ratio != 0) {
            throw std::invalid_argument("SWA packed-byte projection requires K divisible by ratio");
        }
        raw_entries = kernel_block_tokens / desc.compression_ratio;
    }
    if (raw_entries == 0 || raw_entries > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("SWA packed-byte entry count is out of range");
    }
    const uint64_t aligned_entries =
        desc.cp->align_payload.value_or(false) ? alignEntries(raw_entries, cp_size) : raw_entries;
    const uint64_t seq =
        static_cast<uint64_t>(base_block_tokens) * (desc.cp->scale_seq_size.value_or(false) ? cp_size : 1);
    if (aligned_entries > std::numeric_limits<uint32_t>::max() || seq > std::numeric_limits<uint32_t>::max()
        || aligned_entries > std::numeric_limits<size_t>::max() / entry_bytes) {
        throw std::invalid_argument("SWA packed-byte shape exceeds supported range");
    }
    const size_t payload = static_cast<size_t>(aligned_entries) * entry_bytes;
    const bool   sliced  = cp_size > 1 && parallelism.role_type == RoleType::PREFILL;
    size_t       stride  = payload;
    if (sliced) {
        const size_t align = desc.block_stride_bytes_alignment > 0 ?
                                 byteAlignment(desc.block_stride_bytes_alignment, cp_size) :
                                 static_cast<size_t>(cp_size);
        if (payload > std::numeric_limits<size_t>::max() - (align - 1)) {
            throw std::invalid_argument("SWA packed-byte stride overflows size_t");
        }
        stride = (payload + align - 1) / align * align;
    } else if (desc.block_stride_bytes_override > 0) {
        stride = desc.block_stride_bytes_override;
    } else if (desc.block_stride_bytes_alignment > 0 && aligned_entries >= desc.block_stride_alignment_min_entries) {
        const size_t align = desc.block_stride_bytes_alignment;
        if (payload > std::numeric_limits<size_t>::max() - (align - 1)) {
            throw std::invalid_argument("SWA packed-byte stride overflows size_t");
        }
        stride = (payload + align - 1) / align * align;
    }
    if (sliced && stride % cp_size != 0) {
        throw std::invalid_argument("SWA packed-byte stride is not divisible by CP size");
    }
    const size_t  local_stride = sliced ? stride / cp_size : stride;
    const int64_t rank         = sliced ? parallelism.tp_rank : 0;
    if (rank < 0 || static_cast<uint64_t>(rank) >= cp_size) {
        throw std::invalid_argument("SWA packed-byte projection rank exceeds CP size");
    }
    const ResolvedSwaByteLayout value{static_cast<uint32_t>(seq),
                                      static_cast<uint32_t>(aligned_entries),
                                      payload,
                                      stride,
                                      static_cast<size_t>(rank) * local_stride,
                                      local_stride};
    return value;
}

}  // namespace

ResolvedCacheSpecInput resolveCacheSpecInput(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
    ResolvedCacheSpecInput result;
    switch (SpecBuilder::groupType(desc)) {
        case CacheGroupType::FULL:
            result.mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
            break;
        case CacheGroupType::SWA:
            result.mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
            break;
        case CacheGroupType::LINEAR:
            break;
    }
    if (desc.cp) {
        result.mapping = desc.cp->mapping.value_or(result.mapping);
        result.slice   = desc.cp->slice.value_or(result.slice);
    }
    const ParallelismConfig unsharded;
    const auto&             parallelism = ctx.parallelism_config ? *ctx.parallelism_config : unsharded;
    switch (classifySpecProjection(desc)) {
        case Projection::NONE:
            result.physical_layout = NoSpecProjection{};
            break;
        case Projection::STATE:
            result.physical_layout = resolveState(desc, ctx, parallelism);
            break;
        case Projection::SWA:
            result.physical_layout = resolveSwa(desc, ctx, parallelism);
            break;
        case Projection::COMPRESSED:
            result.physical_layout = resolveCompressed(desc, ctx, parallelism);
            break;
    }
    return result;
}

void validateSpecPhysicalLayout(const KVCacheSpecDesc& desc, const SpecPhysicalLayout& layout) {
    RTP_LLM_CHECK_WITH_INFO(!std::holds_alternative<std::monostate>(layout),
                            "cache tag=%s requires a resolved physical spec layout",
                            desc.tag.c_str());
    const auto projection = classifySpecProjection(desc);
    const bool compatible =
        (projection == Projection::NONE && std::holds_alternative<NoSpecProjection>(layout))
        || (projection == Projection::STATE && std::holds_alternative<ResolvedStateEntryLayout>(layout))
        || (projection == Projection::SWA && std::holds_alternative<ResolvedSwaByteLayout>(layout))
        || (projection == Projection::COMPRESSED && std::holds_alternative<ResolvedCompressedLayout>(layout));
    RTP_LLM_CHECK_WITH_INFO(
        compatible, "cache tag=%s has incompatible resolved physical spec layout", desc.tag.c_str());
}

bool sameSpecPhysicalLayout(const SpecPhysicalLayout& lhs, const SpecPhysicalLayout& rhs) {
    if (lhs.index() != rhs.index()) {
        return false;
    }
    if (const auto* a = std::get_if<ResolvedStateEntryLayout>(&lhs)) {
        const auto& b = std::get<ResolvedStateEntryLayout>(rhs);
        return a->seq_size_per_block == b.seq_size_per_block && a->full_entries == b.full_entries
               && a->local_entry_begin == b.local_entry_begin && a->local_entries == b.local_entries
               && a->entry_bytes == b.entry_bytes;
    }
    if (const auto* a = std::get_if<ResolvedSwaByteLayout>(&lhs)) {
        const auto& b = std::get<ResolvedSwaByteLayout>(rhs);
        return a->seq_size_per_block == b.seq_size_per_block && a->full_entries == b.full_entries
               && a->full_payload_bytes == b.full_payload_bytes && a->full_stride_bytes == b.full_stride_bytes
               && a->local_byte_begin == b.local_byte_begin && a->local_stride_bytes == b.local_stride_bytes;
    }
    if (const auto* a = std::get_if<ResolvedCompressedLayout>(&lhs)) {
        const auto& b = std::get<ResolvedCompressedLayout>(rhs);
        return a->seq_size_per_block == b.seq_size_per_block
               && a->entries_per_kernel_block == b.entries_per_kernel_block;
    }
    return true;
}

}  // namespace rtp_llm
