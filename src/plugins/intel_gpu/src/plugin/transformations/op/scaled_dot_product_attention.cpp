// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/scaled_dot_product_attention.hpp"

#include "scaled_dot_product_attention_shape_inference.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

ScaledDotProductAttention::ScaledDotProductAttention(const ov::Output<Node>& Q,
                                                     const ov::Output<Node>& K,
                                                     const ov::Output<Node>& V,
                                                     const ov::Output<Node>& scale,
                                                     const ov::Output<Node>& attn_mask,
                                                     const std::vector<int64_t>& order_q,
                                                     const std::vector<int64_t>& order_k,
                                                     const std::vector<int64_t>& order_v,
                                                     const std::vector<int64_t>& order_out)
    : Op({Q, K, V, scale, attn_mask}),
      m_order_q(order_q),
      m_order_k(order_k),
      m_order_v(order_v),
      m_order_out(order_out) {
    validate_and_infer_types();
}

ScaledDotProductAttention::ScaledDotProductAttention(const ov::Output<Node>& Q,
                                                     const ov::Output<Node>& K,
                                                     const ov::Output<Node>& V,
                                                     const ov::Output<Node>& scale,
                                                     const std::vector<int64_t>& order_q,
                                                     const std::vector<int64_t>& order_k,
                                                     const std::vector<int64_t>& order_v,
                                                     const std::vector<int64_t>& order_out)
    : Op({Q, K, V, scale}),
      m_order_q(order_q),
      m_order_k(order_k),
      m_order_v(order_v),
      m_order_out(order_out) {
    validate_and_infer_types();
}

void ScaledDotProductAttention::validate_and_infer_types() {
    const auto& input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size == 4 || input_size == 5,
                          "Number of inputs is incorrect. Current value is: ",
                          input_size,
                          ", expected 4 or 5.");

    auto out_type = get_input_element_type(0);
    for (size_t i = 1; i < 3; i++) {
        const auto& element_type = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, element_type),
                              "Mixed input types are not supported.");
    }
    NODE_VALIDATION_CHECK(this,
                          out_type.is_real() || out_type.is_dynamic(),
                          "The element type of the input tensor must be a floating-point.");

    ov::op::v13::ScaledDotProductAttention op;
    std::vector<ov::PartialShape> input_shapes = {
        get_input_partial_shape(0),
        get_input_partial_shape(1),
        get_input_partial_shape(2),
    };

    const auto& out_shapes = shape_infer(this, input_shapes, m_order_q, m_order_k, m_order_v, m_order_out);

    set_output_type(0, out_type, out_shapes[0]);
}

bool ScaledDotProductAttention::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<ov::Node> ScaledDotProductAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    switch (new_args.size()) {
    case 4:
        return std::make_shared<ScaledDotProductAttention>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           m_order_q,
                                                           m_order_k,
                                                           m_order_v,
                                                           m_order_out);
    case 5:
        return std::make_shared<ScaledDotProductAttention>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           m_order_q,
                                                           m_order_k,
                                                           m_order_v,
                                                           m_order_out);
    default:
        OPENVINO_THROW("Unable to clone ScaledDotProductAttention ",
                       this->get_friendly_name(),
                       " Incorrect number of inputs. Expected: 4 or 5. Actual: ",
                       new_args.size());
    }
}

std::vector<ov::PartialShape> shape_infer(const ScaledDotProductAttention* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& order_q,
                                          const std::vector<int64_t>& order_k,
                                          const std::vector<int64_t>& order_v,
                                          const std::vector<int64_t>& order_out) {
    auto shape_q = input_shapes[0];
    auto shape_k = input_shapes[1];
    auto shape_v = input_shapes[2];

    // transposed shape
    auto transpose_pshape = [](const ov::PartialShape pshape, const std::vector<int64_t>& order) {
        auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
        for (size_t i = 0; i < order.size(); i++) {
            transposed_pshape[i] = pshape[order[i]];
        }

        return transposed_pshape;
    };

    auto shape_q_t = (order_q.size() > 1) ? transpose_pshape(shape_q, order_q) : shape_q;
    auto shape_k_t = (order_k.size() > 1) ? transpose_pshape(shape_k, order_k) : shape_k;
    auto shape_v_t = (order_v.size() > 1) ? transpose_pshape(shape_v, order_v) : shape_v;

    const auto is_broadcastable = shape_k_t.rank().is_static() && shape_v_t.rank().is_static() &&
                                  ((shape_q_t.size() == shape_k_t.size()) && (shape_q_t.size() == shape_v_t.size()));
    if (is_broadcastable) {
        size_t max_rank = shape_q_t.size();
        for (size_t i = 0; i < max_rank; ++i) {
            if (shape_q_t[i].is_static() && shape_k_t[i].is_static() && shape_v_t[i].is_static()) {
                auto broadcasted_dim = shape_q_t[i].get_length();
                shape_k_t[i] = broadcasted_dim;
                shape_v_t[i] = broadcasted_dim;
            }
        }
    }

    std::vector<ov::PartialShape> transposed_input_shapes{shape_q_t, shape_k_t, shape_v_t};
    for (size_t i = 3; i < transposed_input_shapes.size(); i++) {
        transposed_input_shapes.push_back(input_shapes[i]);
    }

    OPENVINO_ASSERT(op != nullptr, "op should not be nullptr for shape_infer.");
    auto op_v13 = dynamic_cast<const ov::op::v13::ScaledDotProductAttention*>(op);
    OPENVINO_ASSERT(op_v13 != nullptr, "ov::op::v13::ScaledDotProductAttention*>(op) should not be nullptr.");
    auto out_shapes = ov::op::v13::shape_infer(op_v13, transposed_input_shapes);

    if (order_out.size() > 0) {
        return {transpose_pshape(out_shapes[0], order_out)};
    } else {
        return {out_shapes[0]};
    }
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
