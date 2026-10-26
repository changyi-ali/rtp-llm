#include "rtp_llm/cpp/cache/CacheSpecLayoutResolver.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <limits>

namespace rtp_llm {
namespace {
template<typename T, typename = void>
struct HasPublicSpecBuild: std::false_type {};

template<typename T>
struct HasPublicSpecBuild<T,
                          std::void_t<decltype(T::build(std::declval<const KVCacheSpecDesc&>(),
                                                        std::declval<const SpecBuildContext&>(),
                                                        std::declval<const SpecPhysicalLayout&>()))>>:
    std::true_type {};

static_assert(!HasPublicSpecBuild<MHAKVCacheSpec>::value);
static_assert(!HasPublicSpecBuild<MLAKVCacheSpec>::value);
static_assert(!HasPublicSpecBuild<LinearKVCacheSpec>::value);
static_assert(!HasPublicSpecBuild<CompressedKVCacheSpec>::value);
static_assert(!HasPublicSpecBuild<FixedStateCacheSpec>::value);

KVCacheSpecDesc stateDesc() {
    KVCacheSpecDesc desc;
    desc.tag                                  = "state";
    desc.cache_type                           = KVCacheSpecType::OpaqueState;
    desc.entry_elems                          = 4;
    desc.entry_dtype                          = DataType::TYPE_FP32;
    desc.entry_count_mode                     = OpaqueBlockEntryCountMode::STATE_RING;
    desc.compression_ratio                    = 4;
    desc.state_ring_overlap                   = 1;
    desc.state_ring_include_gen_num_per_cycle = true;
    desc.cp.emplace();
    desc.cp->scale_seq_size       = true;
    desc.cp->align_payload        = true;
    desc.cp->prefill_slice_layout = CpPrefillSliceLayout::PAYLOAD;
    return desc;
}
KVCacheSpecDesc swaDesc() {
    auto desc                         = stateDesc();
    desc.tag                          = "swa";
    desc.entry_count_mode             = OpaqueBlockEntryCountMode::EXPLICIT;
    desc.explicit_entry_count         = 128;
    desc.entry_elems                  = 584;
    desc.entry_dtype                  = DataType::TYPE_UINT8;
    desc.block_stride_bytes_alignment = 576;
    desc.cp->align_payload            = false;
    desc.cp->prefill_slice_layout     = CpPrefillSliceLayout::BLOCK_STRIDE;
    return desc;
}
KVCacheSpecDesc compressedDesc() {
    auto desc                     = stateDesc();
    desc.tag                      = "compressed";
    desc.cache_type               = KVCacheSpecType::OpaqueKV;
    desc.entry_count_mode         = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.cp->scale_seq_size       = false;
    desc.cp->align_payload        = false;
    desc.cp->prefill_slice_layout = CpPrefillSliceLayout::PAYLOAD;
    return desc;
}
class CacheSpecLayoutResolverTest: public ::testing::Test {
protected:
    ParallelismConfig parallelism;
    SpecBuildContext  ctx;
    void              SetUp() override {
        parallelism.role_type                          = RoleType::PREFILL;
        parallelism.tp_size                            = 4;
        parallelism.tp_rank                            = 3;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        ctx.parallelism_config                         = &parallelism;
        ctx.seq_size_per_block                         = 256;
        ctx.kernel_tokens_per_block                    = 128;
        ctx.gen_num_per_cycle                          = 2;
    }
    void rejects(const KVCacheSpecDesc& desc, const std::string& message) {
        try {
            resolveCacheSpecInput(desc, ctx);
            FAIL() << "accepted malformed descriptor: " << message;
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
        }
    }
    void matches(const KVCacheSpecDesc& desc, const SpecPhysicalLayout& expected) {
        const auto actual = resolveCacheSpecInput(desc, ctx);
        EXPECT_TRUE(sameSpecPhysicalLayout(actual.physical_layout, expected));
        EXPECT_NO_THROW(validateSpecPhysicalLayout(desc, actual.physical_layout));
    }
};

TEST_F(CacheSpecLayoutResolverTest, ValidNoProjectionAndPolicyDefaults) {
    for (const auto type : {KVCacheSpecType::MultiHeadAttention,
                            KVCacheSpecType::MultiHeadLatentAttention,
                            KVCacheSpecType::LinearAttention,
                            KVCacheSpecType::OpaqueState,
                            KVCacheSpecType::OpaqueKV}) {
        KVCacheSpecDesc desc;
        desc.tag        = "ordinary";
        desc.cache_type = type;
        for (int mode = 0; mode < 3; ++mode) {
            if (mode > 0)
                desc.cp.emplace();
            if (mode == 2) {
                desc.cp->mapping = CpBlockMappingMode::NONE;
                desc.cp->slice   = CpBlockSliceMode::PAYLOAD_BYTES;
            }
            const auto input = resolveCacheSpecInput(desc, ctx);
            EXPECT_TRUE(std::holds_alternative<NoSpecProjection>(input.physical_layout));
            const auto mapping = type == KVCacheSpecType::LinearAttention ? CpBlockMappingMode::NONE :
                                 type == KVCacheSpecType::OpaqueState     ? CpBlockMappingMode::COMPACT_LAST_RANK :
                                                                            CpBlockMappingMode::BLOCK_ROUND_ROBIN;
            EXPECT_EQ(input.mapping, mode == 2 ? CpBlockMappingMode::NONE : mapping);
            EXPECT_EQ(input.slice, mode == 2 ? CpBlockSliceMode::PAYLOAD_BYTES : CpBlockSliceMode::NONE);
        }
    }
}

TEST_F(CacheSpecLayoutResolverTest, ValidStatePrefillCp1248) {
    const auto desc = stateDesc();
    for (const uint32_t cp : {1u, 2u, 4u, 8u}) {
        parallelism.tp_size                 = cp;
        parallelism.tp_rank                 = cp - 1;
        const uint32_t                 full = cp == 8 ? 16 : cp == 4 ? 12 : 10;
        const ResolvedStateEntryLayout expected{256 * cp, full, (cp - 1) * (full / cp), full / cp, 16};
        matches(desc, expected);
    }
}

TEST_F(CacheSpecLayoutResolverTest, ValidSwaPackedBytes) {
    const auto desc = swaDesc();
    for (const uint32_t cp : {1u, 2u, 4u, 8u}) {
        parallelism.tp_size = cp;
        parallelism.tp_rank = cp - 1;
        const ResolvedSwaByteLayout expected{256 * cp, 128, 74752, 74880, (cp - 1) * (74880 / cp), 74880 / cp};
        matches(desc, expected);
    }
}

TEST_F(CacheSpecLayoutResolverTest, ValidCompressedIndependentKernelPages) {
    const auto desc = compressedDesc();
    for (const uint32_t cp : {1u, 2u, 4u, 8u}) {
        parallelism.tp_size = cp;
        const ResolvedCompressedLayout expected{256, 32 / cp};
        matches(desc, expected);
        EXPECT_EQ(expected.seq_size_per_block / SpecBuilder::kernelSeqSizePerBlock(desc, ctx, 256), 2u);
    }
}

TEST_F(CacheSpecLayoutResolverTest, ValidExplicitProjectionUnshardedAndMissingParallelism) {
    parallelism.prefill_cp_config.kv_cache_sharded = false;
    for (const bool missing : {false, true}) {
        ctx.parallelism_config = missing ? nullptr : &parallelism;
        matches(stateDesc(), ResolvedStateEntryLayout{256, 10, 0, 10, 16});
        matches(swaDesc(), ResolvedSwaByteLayout{256, 128, 74752, 74880, 0, 74880});
        matches(compressedDesc(), ResolvedCompressedLayout{256, 32});
    }
}

TEST_F(CacheSpecLayoutResolverTest, ValidDecodeUsesProducerCp8WithLocalTp1) {
    parallelism.role_type                         = RoleType::DECODE;
    parallelism.tp_size                           = 1;
    parallelism.tp_rank                           = 0;
    parallelism.prefill_cp_config.method          = CPRotateMethod::PREFILL_CP;
    parallelism.prefill_cp_config.prefill_cp_size = 8;
    matches(stateDesc(), ResolvedStateEntryLayout{2048, 16, 0, 16, 16});
    matches(swaDesc(), ResolvedSwaByteLayout{2048, 128, 74752, 74880, 0, 74880});
    auto desc               = compressedDesc();
    desc.cp->scale_seq_size = true;
    matches(desc, ResolvedCompressedLayout{2048, 32});
}

TEST_F(CacheSpecLayoutResolverTest, ValidAllFieldsParticipateInEquality) {
    const ResolvedStateEntryLayout state{1024, 12, 9, 3, 16};
    for (auto member : {&ResolvedStateEntryLayout::seq_size_per_block,
                        &ResolvedStateEntryLayout::full_entries,
                        &ResolvedStateEntryLayout::local_entry_begin,
                        &ResolvedStateEntryLayout::local_entries}) {
        auto changed = state;
        ++(changed.*member);
        EXPECT_FALSE(sameSpecPhysicalLayout(state, changed));
    }
    auto state_bytes = state;
    ++state_bytes.entry_bytes;
    EXPECT_FALSE(sameSpecPhysicalLayout(state, state_bytes));
    const ResolvedSwaByteLayout swa{1024, 128, 74752, 74880, 56160, 18720};
    for (auto member : {&ResolvedSwaByteLayout::seq_size_per_block, &ResolvedSwaByteLayout::full_entries}) {
        auto changed = swa;
        ++(changed.*member);
        EXPECT_FALSE(sameSpecPhysicalLayout(swa, changed));
    }
    for (auto member : {&ResolvedSwaByteLayout::full_payload_bytes,
                        &ResolvedSwaByteLayout::full_stride_bytes,
                        &ResolvedSwaByteLayout::local_byte_begin,
                        &ResolvedSwaByteLayout::local_stride_bytes}) {
        auto changed = swa;
        ++(changed.*member);
        EXPECT_FALSE(sameSpecPhysicalLayout(swa, changed));
    }
    const ResolvedCompressedLayout compressed{256, 8};
    for (auto member :
         {&ResolvedCompressedLayout::seq_size_per_block, &ResolvedCompressedLayout::entries_per_kernel_block}) {
        auto changed = compressed;
        ++(changed.*member);
        EXPECT_FALSE(sameSpecPhysicalLayout(compressed, changed));
    }
    EXPECT_FALSE(sameSpecPhysicalLayout(state, swa));
}

TEST_F(CacheSpecLayoutResolverTest, RejectsZeroBaseSpan) {
    ctx.seq_size_per_block = 0;
    rejects(stateDesc(), "STATE layout requires positive base block tokens");
    rejects(swaDesc(), "SWA layout requires positive base block tokens");
    rejects(compressedDesc(), "compressed layout requires positive base block tokens");
}

TEST_F(CacheSpecLayoutResolverTest, RejectsInvalidEntryFormatAndCounts) {
    for (auto desc : {stateDesc(), swaDesc()}) {
        const auto original = desc;
        desc.entry_elems    = 0;
        rejects(desc, "requires a valid entry format");
        desc             = original;
        desc.entry_dtype = DataType::TYPE_INVALID;
        rejects(desc, "requires a valid entry format");
        desc.entry_dtype = static_cast<DataType>(127);
        rejects(desc, "requires a valid entry format");
    }
    auto state             = stateDesc();
    state.entry_count_mode = OpaqueBlockEntryCountMode::EXPLICIT;
    rejects(state, "requires positive entries");
    auto swa                 = swaDesc();
    swa.explicit_entry_count = 0;
    rejects(swa, "entry count is out of range");
}

TEST_F(CacheSpecLayoutResolverTest, RejectsInvalidSliceAndTypedInputs) {
    auto desc                     = stateDesc();
    desc.cp->prefill_slice_layout = static_cast<CpPrefillSliceLayout>(99);
    rejects(desc, "cache tag=state has invalid prefill slice layout");
    for (const auto& typed_desc : {stateDesc(), swaDesc(), compressedDesc()}) {
        EXPECT_THROW(validateSpecPhysicalLayout(typed_desc, std::monostate{}), std::exception);
        EXPECT_THROW(validateSpecPhysicalLayout(typed_desc, NoSpecProjection{}), std::exception);
    }
    EXPECT_THROW(validateSpecPhysicalLayout(stateDesc(), ResolvedSwaByteLayout{}), std::exception);
    EXPECT_THROW(validateSpecPhysicalLayout(swaDesc(), ResolvedStateEntryLayout{}), std::exception);
    EXPECT_THROW(validateSpecPhysicalLayout(compressedDesc(), ResolvedStateEntryLayout{}), std::exception);
}

TEST_F(CacheSpecLayoutResolverTest, RejectsZeroAndNonDivisibleRatios) {
    for (auto desc : {stateDesc(), swaDesc(), compressedDesc()}) {
        desc.entry_count_mode  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
        desc.compression_ratio = 0;
        rejects(desc, desc.tag == "swa" ? "requires a valid entry format" : "requires K divisible by ratio");
        desc.compression_ratio = 3;
        rejects(desc, "requires K divisible by ratio");
    }
}

TEST_F(CacheSpecLayoutResolverTest, RejectsNegativeAndOutOfRangeRank) {
    for (const int64_t rank : {-1LL, 4LL, 4294967296LL}) {
        parallelism.tp_rank = rank;
        rejects(stateDesc(), "STATE entry projection rank exceeds CP size");
        rejects(swaDesc(), "SWA packed-byte projection rank exceeds CP size");
    }
}

TEST_F(CacheSpecLayoutResolverTest, RejectsMissingProducerCpAndOverflowingCp) {
    parallelism.role_type                = RoleType::DECODE;
    parallelism.tp_size                  = 1;
    parallelism.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    for (const int cp : {0, 1}) {
        parallelism.prefill_cp_config.prefill_cp_size = cp;
        for (const auto& desc : {stateDesc(), swaDesc(), compressedDesc()}) {
            rejects(desc, "fixed region decode CP layout requires explicit prefill_cp_size");
        }
    }
    parallelism.prefill_cp_config.prefill_cp_size = int64_t{1} << 32;
    rejects(stateDesc(), "fixed region CP size exceeds uint32 range");
}

TEST_F(CacheSpecLayoutResolverTest, RejectsSpanRingAndEntryOverflow) {
    ctx.seq_size_per_block = std::numeric_limits<uint32_t>::max();
    rejects(stateDesc(), "STATE entry projection exceeds uint32 range");
    rejects(swaDesc(), "SWA packed-byte shape exceeds supported range");
    auto compressed               = compressedDesc();
    compressed.cp->scale_seq_size = true;
    rejects(compressed, "compressed CP block token span exceeds uint32 range");
    ctx.seq_size_per_block = 256;
    for (auto desc : {stateDesc(), swaDesc(), compressedDesc()}) {
        desc.entry_count_mode   = OpaqueBlockEntryCountMode::STATE_RING;
        desc.state_ring_overlap = std::numeric_limits<uint32_t>::max();
        desc.compression_ratio  = std::numeric_limits<uint32_t>::max();
        ctx.gen_num_per_cycle   = std::numeric_limits<uint32_t>::max();
        rejects(desc, "fixed region ring entry count overflows uint64");
    }
}

TEST_F(CacheSpecLayoutResolverTest, RejectsByteLcmPayloadAndStrideOverflow) {
    auto desc                         = swaDesc();
    desc.block_stride_bytes_alignment = std::numeric_limits<size_t>::max();
    rejects(desc, "SWA packed-byte stride overflows size_t");
    parallelism.tp_size = 1;
    parallelism.tp_rank = 0;
    rejects(desc, "SWA packed-byte stride overflows size_t");
    desc.block_stride_bytes_alignment = 1;
    desc.entry_dtype                  = DataType::TYPE_FP64;
    desc.entry_elems                  = std::numeric_limits<uint32_t>::max();
    desc.explicit_entry_count         = std::numeric_limits<uint32_t>::max();
    rejects(desc, "SWA packed-byte shape exceeds supported range");
}
class SpecBuildInputTest: public CacheSpecLayoutResolverTest {};

TEST_F(SpecBuildInputTest, RejectsUnresolvedAndIncompatibleFamilies) {
    KVCacheSpecDesc ordinary;
    ordinary.tag = "ordinary";
    for (const auto& desc : {ordinary, stateDesc(), swaDesc(), compressedDesc()}) {
        const auto                            resolved = resolveCacheSpecInput(desc, ctx);
        const std::vector<SpecPhysicalLayout> inputs{std::monostate{},
                                                     NoSpecProjection{},
                                                     ResolvedStateEntryLayout{},
                                                     ResolvedSwaByteLayout{},
                                                     ResolvedCompressedLayout{}};
        for (const auto& input : inputs) {
            if (input.index() == resolved.physical_layout.index()) {
                continue;
            }
            const std::string message = "cache tag=" + desc.tag
                                        + (input.index() == 0 ? " requires a resolved physical spec layout" :
                                                                " has incompatible resolved physical spec layout");
            try {
                SpecBuilder::build(desc, ctx, input);
                FAIL() << "accepted incompatible input for " << desc.tag;
            } catch (const std::exception& error) {
                EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
            }
        }
    }
}

TEST_F(SpecBuildInputTest, BuildsExplicitNoProjectionForAllSpecTypes) {
    AttentionConfigs attention;
    attention.kv_head_num   = 4;
    attention.size_per_head = 2;
    attention.kv_lora_rank  = 4;
    attention.rope_head_dim = 2;
    LinearAttentionConfig linear;
    linear.linear_num_key_heads   = 4;
    linear.linear_num_value_heads = 4;
    linear.linear_key_head_dim    = 2;
    linear.linear_value_head_dim  = 2;
    linear.linear_conv_kernel_dim = 4;
    linear.ssm_state_dtype        = DataType::TYPE_FP32;
    linear.conv_state_dtype       = DataType::TYPE_FP32;
    ctx.attn_config               = &attention;
    ctx.linear_attention_config   = &linear;
    ctx.dtype                     = DataType::TYPE_FP32;
    for (const auto type : {KVCacheSpecType::MultiHeadAttention,
                            KVCacheSpecType::MultiHeadLatentAttention,
                            KVCacheSpecType::LinearAttention,
                            KVCacheSpecType::OpaqueKV,
                            KVCacheSpecType::OpaqueState}) {
        KVCacheSpecDesc desc;
        desc.tag                  = "ordinary";
        desc.cache_type           = type;
        desc.entry_dtype          = DataType::TYPE_FP32;
        desc.entry_elems          = 2;
        desc.explicit_entry_count = 4;
        const auto input          = resolveCacheSpecInput(desc, ctx);
        ASSERT_TRUE(std::holds_alternative<NoSpecProjection>(input.physical_layout));
        const auto spec = SpecBuilder::build(desc, ctx, input.physical_layout);
        ASSERT_NE(spec, nullptr);
        EXPECT_EQ(spec->type, type);
        EXPECT_EQ(spec->seq_size_per_block, 256u);
        EXPECT_GT(spec->block_size_bytes(), 0u);
    }
}

TEST_F(SpecBuildInputTest, BuildsResolvedPhysicalValues) {
    for (const auto& desc : {stateDesc(), swaDesc(), compressedDesc()}) {
        const auto input = resolveCacheSpecInput(desc, ctx);
        const auto spec  = SpecBuilder::build(desc, ctx, input.physical_layout);
        ASSERT_NE(spec, nullptr);
        EXPECT_EQ(spec->seq_size_per_block, desc.tag == "compressed" ? 256u : 1024u);
        EXPECT_EQ(spec->block_size_bytes(), desc.tag == "compressed" ? 256u : desc.tag == "swa" ? 18720u : 48u);
    }
}
}  // namespace
}  // namespace rtp_llm
