// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "scaled_dot_product_attention_inst.h"
#include "scaled_dot_product_attention_shape_inference.hpp"
#include "utils.hpp"

#include "intel_gpu/op/scaled_dot_product_attention.hpp"


namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scaled_dot_product_attention)

layout scaled_dot_product_attention_inst::calc_output_layout(scaled_dot_product_attention_node const& node,
                                                             kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<scaled_dot_product_attention>();
    auto transpose_shape = [&desc](const ov::PartialShape& shape, const std::vector<int64_t>& order) {
        if (desc->input_q_transpose_order.empty())
            return shape;

        auto shape_transposed = ov::PartialShape(shape);
        auto rank_diff = shape.size() - order.size();
        for (size_t i = 0; i < order.size(); i++) {
            size_t idx = static_cast<size_t>(order[i]);
            shape_transposed[i + rank_diff] = shape[idx + rank_diff];
        }

        return shape_transposed;
    };

    auto input0_layout = impl_param.get_input_layout(0);
    auto default_out_dt =
        data_type_traits::is_floating_point(input0_layout.data_type) ? input0_layout.data_type : data_types::f32;
    auto output_type = desc->output_data_types[0].value_or(default_out_dt);
    auto output_format = input0_layout.format;
    auto output_shape = transpose_shape(input0_layout.get_partial_shape(),
                                        desc->input_q_transpose_order);  // output shape matches with Q input shape

    return layout{output_shape, output_type, output_format, desc->output_paddings[0]};
}

template <typename ShapeType>
std::vector<layout> scaled_dot_product_attention_inst::calc_output_layouts(
    scaled_dot_product_attention_node const& node,
    kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.typed_desc<scaled_dot_product_attention>()->output_data_types[0]) == false &&
           "Output data type forcing is not supported for reshape_node!");

    // TBD: broadcasting
    auto prim = impl_param.typed_desc<scaled_dot_product_attention>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto default_out_dt =
        data_type_traits::is_floating_point(input0_layout.data_type) ? input0_layout.data_type : data_types::f32;
    auto output_type = prim->output_data_types[0].value_or(default_out_dt);

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ov::intel_gpu::op::ScaledDotProductAttention op;

    std::vector<ShapeType> input_shapes;
    for (size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(0).get<ShapeType>());
    }

    std::vector<ShapeType> output_shapes = ov::intel_gpu::op::shape_infer(&op,
                                                                          input_shapes,
                                                                          prim->input_q_transpose_order,
                                                                          prim->input_k_transpose_order,
                                                                          prim->input_v_transpose_order,
                                                                          prim->output_transpose_order);
    cldnn::format output_format = input0_layout.format;

    return {layout{output_shapes[0], output_type, output_format, prim->output_paddings[0]}};
}

template std::vector<layout> scaled_dot_product_attention_inst::calc_output_layouts<ov::PartialShape>(
    scaled_dot_product_attention_node const& node,
    const kernel_impl_params& impl_param);

std::string scaled_dot_product_attention_inst::to_string(scaled_dot_product_attention_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

scaled_dot_product_attention_inst::typed_primitive_inst(network& network, scaled_dot_product_attention_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
