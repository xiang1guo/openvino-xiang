// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/sdpa_onednn.hpp"
#endif

// TODO: regist the ocl sdpa implementation here.
// 1. Add implementation manager for OCL sdpa implementation
// 2. Regist here

// #if OV_GPU_WITH_OCL
// #    include "impls/ocl/scaled_dot_product_attention.hpp"
// #endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<ScaledDotProductAttention>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::ScaledDotProductAttentionImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ScaledDotProductAttentionImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ScaledDotProductAttentionImplementationManager, shape_types::dynamic_shape,
            [](const cldnn::program_node& node){
                if (node.can_use(impl_types::onednn))
                    return false;
                return node.as<ScaledDotProductAttention>().use_explicit_padding();
        })
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
