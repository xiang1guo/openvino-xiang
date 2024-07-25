// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_dot_product_attention_partial_decomposition.hpp"

#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/logical_not.hpp"
// #include "openvino/op/matmul.hpp"
#include "intel_gpu/op/scaled_dot_product_attention.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/any.hpp"

// TODO: remove this by adding order attrs in graph API sdpa
#include "intel_gpu/op/sdpa.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov {
namespace intel_gpu {

ScaledDotProductAttentionPartialDecomposition::ScaledDotProductAttentionPartialDecomposition() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
        case ov::element::f16:
        case ov::element::f32:
            return true;
        default:
            return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr &&
               is_fp_type(output);
    };

    auto input_q_m = any_input(not_transpose);
    auto input_k_m = any_input(not_transpose);
    auto input_v_m = any_input(not_transpose);
    auto input_attn_mask = any_input(not_transpose);
    auto input_scale = any_input(not_transpose);
    auto transpose_q_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_k_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_v_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_fp_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_fp_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_fp_type);

    auto sdpa_in_q = std::make_shared<Or>(OutputVector{input_q_m, transpose_q_m});
    auto sdpa_in_k = std::make_shared<Or>(OutputVector{input_k_m, transpose_k_m});
    auto sdpa_in_v = std::make_shared<Or>(OutputVector{input_v_m, transpose_v_m});

    auto sdpa_without_attn_mask_m =
        wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v});
    auto sdpa_with_attn_mask_m =
        wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask});
    auto sdpa_with_attn_mask_and_scale_m = wrap_type<ov::op::v13::ScaledDotProductAttention>(
        {sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask, input_scale});
    auto sdpa_m = std::make_shared<Or>(
        OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());

        if (!sdpa || transformation_callback(sdpa)) {
            return false;
        }

        // TODO: need to add these part in graph internal sdpa
        auto order_q = op::SDPA::default_order(sdpa->get_input_partial_shape(0).size());
        auto order_k = op::SDPA::default_order(sdpa->get_input_partial_shape(1).size());
        auto order_v = op::SDPA::default_order(sdpa->get_input_partial_shape(2).size());
        auto order_output = op::SDPA::default_order(sdpa->get_output_partial_shape(0).size());

        size_t input_q_output_idx = sdpa->get_input_source_output(0).get_index();
        size_t input_k_output_idx = sdpa->get_input_source_output(1).get_index();
        size_t input_v_output_idx = sdpa->get_input_source_output(2).get_index();

        auto process_transpose = [](const std::shared_ptr<Node>& transpose_node,
                                    const std::shared_ptr<Node>& transpose_order_const_node,
                                    std::vector<int64_t>& order,
                                    size_t& output_idx) {
            auto transpose_order_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose_order_const_node);

            order = transpose_order_const->cast_vector<int64_t>();
            // Allow any transposes without head_size dim position change
            if (order.back() != static_cast<int64_t>(order.size() - 1))
                return false;

            auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            return true;
        };

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q,
                                                     input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k,
                                                     input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v,
                                                     input_v_output_idx);

        if (!can_fuse_transposes)
            return false;

        auto input_q = ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx);
        auto input_k = ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx);
        auto input_v = ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx);

        auto new_output_node = decompose(sdpa, input_q, input_k, input_v, order_q, order_k, order_v, order_output);
        ov::replace_node(sdpa, new_output_node);
        return true;
    };

    auto m =
        std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "ScaledDotProductAttentionPartialDecomposition");
    register_matcher(m, callback);
}

std::shared_ptr<ov::Node> ScaledDotProductAttentionPartialDecomposition::decompose(
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node,
    ov::Output<Node>& query,
    ov::Output<Node>& key,
    ov::Output<Node>& value,
    const std::vector<int64_t>& order_q,
    const std::vector<int64_t>& order_k,
    const std::vector<int64_t>& order_v,
    const std::vector<int64_t>& order_out) {
    using namespace ov::op;
    // auto query = node->input_value(0);
    // auto key = node->input_value(1);
    // auto value = node->input_value(2);

    auto q_shape = register_new_node<v3::ShapeOf>(query, element::i32);
    auto k_shape = register_new_node<v3::ShapeOf>(key, element::i32);
    auto minus_one = register_new_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto minus_two = register_new_node(v0::Constant::create(element::i32, Shape{}, {-2}));
    auto minus_inf =
        register_new_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}));
    auto zero_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto one_f = register_new_node<v1::ConvertLike>(one_i, query);
    auto zero_f = register_new_node<v1::ConvertLike>(zero_i, query);

    Output<Node> scale;
    if (node->get_input_size() < 5) {
        scale = register_new_node<v8::Gather>(q_shape, minus_one, zero_i)->output(0);
        scale = register_new_node<v1::ConvertLike>(scale, query);
        auto sqrt_scale = register_new_node<v0::Sqrt>(scale);
        scale = register_new_node<v1::Divide>(one_f, sqrt_scale);
    } else {
        scale = node->input_value(4);
    }

    std::shared_ptr<op::ScaledDotProductAttention> gpu_sdpa;
    if (node->get_causal() || node->get_input_size() > 3) {
        Output<Node> mask;
        Output<Node> atten_mask;
        if (!node->get_causal()) {
            mask = node->input_value(3);

            // two types of masks are supported. A boolean mask where a value of True indicates that the element should
            // take part in attention. A float mask of the same type as query, key, value that is added to the attention
            // score.
            if (mask.get_element_type() == element::boolean) {
                atten_mask = register_new_node<v1::ConvertLike>(mask, query);
                auto inv_mask = register_new_node<v1::LogicalNot>(mask);
                atten_mask = register_new_node<v1::Select>(inv_mask, atten_mask, minus_inf);
            } else {
                atten_mask = mask;
            }
        } else {
            auto target_s_len = register_new_node<v8::Gather>(q_shape, minus_two, zero_i);
            auto source_s_len = register_new_node<v8::Gather>(k_shape, minus_two, zero_i);
            auto ssl = register_new_node<v0::Unsqueeze>(source_s_len, zero_i);
            auto tsl = register_new_node<v0::Unsqueeze>(target_s_len, zero_i);
            auto mask_shape = register_new_node<v0::Concat>(OutputVector{tsl, ssl}, 0);
            mask = register_new_node<v1::Broadcast>(minus_inf, mask_shape);
            auto horizontal_range = register_new_node<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
            horizontal_range = register_new_node<v0::Unsqueeze>(horizontal_range, zero_i);
            auto stop = register_new_node<v1::Add>(target_s_len, one_i);
            auto vertical_range = register_new_node<v4::Range>(one_i, stop, one_i, element::i32)->output(0);
            vertical_range = register_new_node<v0::Unsqueeze>(vertical_range, one_i);
            auto triu = register_new_node<v1::GreaterEqual>(horizontal_range, vertical_range);
            atten_mask = register_new_node<v1::Select>(triu, mask, zero_f);
        }
        gpu_sdpa = register_new_node<op::ScaledDotProductAttention>(query, key, value, scale, atten_mask, order_q, order_k, order_v, order_out);
    } else {
        gpu_sdpa = register_new_node<op::ScaledDotProductAttention>(query, key, value, scale, order_q, order_k, order_v, order_out);
    }

    gpu_sdpa->set_friendly_name(node->get_friendly_name());
    copy_runtime_info(node, get_new_nodes());
    return gpu_sdpa;
}

}  // namespace intel_gpu
}  // namespace ov
