// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace cldnn {
namespace onednn {
void register_implementations();

namespace detail {

#define REGISTER_ONEDNN_IMPL(prim)  \
    struct attach_##prim##_onednn { \
        attach_##prim##_onednn();   \
    }

REGISTER_ONEDNN_IMPL(convolution);
REGISTER_ONEDNN_IMPL(deconvolution);
REGISTER_ONEDNN_IMPL(concatenation);
REGISTER_ONEDNN_IMPL(gemm);
REGISTER_ONEDNN_IMPL(pooling);
REGISTER_ONEDNN_IMPL(reduction);
REGISTER_ONEDNN_IMPL(reorder);
REGISTER_ONEDNN_IMPL(fully_connected);
REGISTER_ONEDNN_IMPL(scaled_dot_product_attention_graph);

#undef REGISTER_ONEDNN_IMPL

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
