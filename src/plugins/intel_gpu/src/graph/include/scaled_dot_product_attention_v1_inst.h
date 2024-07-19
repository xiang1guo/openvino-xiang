// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/scaled_dot_product_attention_v1.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<scaled_dot_product_attention_v1> : public typed_program_node_base<scaled_dot_product_attention_v1> {
    using parent = typed_program_node_base<scaled_dot_product_attention_v1>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using scaled_dot_product_attention_v1_node = typed_program_node<scaled_dot_product_attention_v1>;

template <>
class typed_primitive_inst<scaled_dot_product_attention_v1> : public typed_primitive_inst_base<scaled_dot_product_attention_v1> {
    using parent = typed_primitive_inst_base<scaled_dot_product_attention_v1>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(scaled_dot_product_attention_v1_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(scaled_dot_product_attention_v1_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(scaled_dot_product_attention_v1_node const& node);
    bool has_indirect_inputs() const {
        return get_typed_desc<scaled_dot_product_attention_v1>()->indirect_axis != -1;
    }

    typed_primitive_inst(network& network, scaled_dot_product_attention_v1_node const& desc);
};

using scaled_dot_product_attention_v1_inst = typed_primitive_inst<scaled_dot_product_attention_v1>;
}  // namespace cldnn
