#pragma once

#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {

struct ResolvedCacheSpecInput {
    SpecPhysicalLayout physical_layout;
    CpBlockMappingMode mapping = CpBlockMappingMode::NONE;
    CpBlockSliceMode   slice   = CpBlockSliceMode::NONE;
};

ResolvedCacheSpecInput resolveCacheSpecInput(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx);
void                   validateSpecPhysicalLayout(const KVCacheSpecDesc& desc, const SpecPhysicalLayout& layout);

}  // namespace rtp_llm
