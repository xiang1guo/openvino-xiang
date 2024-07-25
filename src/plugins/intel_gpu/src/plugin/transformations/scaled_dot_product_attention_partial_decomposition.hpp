// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gpu {

class ScaledDotProductAttentionPartialDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ScaledDotProductAttentionPartialDecomposition", "0");
    ScaledDotProductAttentionPartialDecomposition();
    std::shared_ptr<ov::Node> decompose(std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node,
    ov::Output<Node>& query,
    ov::Output<Node>& key,
    ov::Output<Node>& value,
    const std::vector<int64_t>& order_q,
    const std::vector<int64_t>& order_k,
    const std::vector<int64_t>& order_v,
    const std::vector<int64_t>& order_out);
};

}  // namespace intel_gpu
}  // namespace ov
