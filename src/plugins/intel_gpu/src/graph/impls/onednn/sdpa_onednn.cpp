// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <oneapi/dnnl/dnnl_graph.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include <ostream>

#include "primitive_onednn_graph_base.hpp"
#include "impls/registry/implementation_manager.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_onednn.hpp"
using namespace dnnl::graph;
namespace cldnn {
namespace onednn {
struct scaled_dot_product_attention_graph_onednn : typed_primitive_onednn_graph_impl<scaled_dot_product_attention> {
    using parent = typed_primitive_onednn_graph_impl<scaled_dot_product_attention>;
    using parent::parent;
    bool _has_attn_mask;
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::scaled_dot_product_attention_graph_onednn)
    scaled_dot_product_attention_graph_onednn(const engine& engine,
                                              const ExecutionConfig& config,
                                              const kernel_params_t& sdpa_param,
                                              const std::vector<cldnn::layout>& input_layouts,
                                              const cldnn::layout& output_layout)
        : parent(engine, config, input_layouts, output_layout),
          _has_attn_mask(_input_layouts.size() == 5) {
        build_graph(engine, sdpa_param);
    }
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);
        ob << _has_attn_mask;
#endif
    }
    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);
        ib >> _has_attn_mask;
        // build_graph(ib.get_engine());
#endif
    }
    void build_graph(const engine& engine, const kernel_params_t& sdpa_param) {
        auto dnnl_engine = engine.get_onednn_engine();
        auto dtype = logical_tensor::data_type::f16;
        auto query_shape = get_logical_tensor_dims(_input_layouts[0], sdpa_param.input0_order);
        auto key_shape = get_logical_tensor_dims(_input_layouts[1], sdpa_param.input1_order);
        auto value_shape = get_logical_tensor_dims(_input_layouts[2], sdpa_param.input2_order);
        bool has_attn_mask = _input_layouts.size() > 4;
        int64_t batch_size = query_shape[0];
        int64_t num_head = query_shape[1];
        int64_t seq_len = query_shape[2];
        logical_tensor::dims qk_output_shape = {batch_size, num_head, seq_len, key_shape[2]};
        logical_tensor::dims scale_shape = {1};
        size_t lt_id = 5;
        logical_tensor query_input{0, dtype, query_shape, logical_tensor::layout_type::strided};
        logical_tensor key_input{1, dtype, key_shape, logical_tensor::layout_type::strided};
        logical_tensor matmul_qk_out{lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
        op matmul_qk{0, op::kind::MatMul, {query_input, key_input}, {matmul_qk_out}, "matmul_qk"};
        matmul_qk.set_attr<bool>(op::attr::transpose_b, true);
        logical_tensor scale_factor{3,
                                    dtype,
                                    scale_shape,
                                    logical_tensor::layout_type::strided,
                                    logical_tensor::property_type::constant};
        logical_tensor scaled_qk_out{lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
        op scale_div{1, op::kind::Multiply, {matmul_qk_out, scale_factor}, {scaled_qk_out}, "scale_div"};
        std::shared_ptr<graph> g;
        if (has_attn_mask) {
            auto attn_mask_shape = get_logical_tensor_dims(_input_layouts[4]);
            logical_tensor attention_mask{4, dtype, attn_mask_shape, logical_tensor::layout_type::strided};
            logical_tensor masked_qk_out{lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
            op mask_add{2, op::kind::Add, {scaled_qk_out, attention_mask}, {masked_qk_out}, "mask_add"};
            logical_tensor softmax_out{lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
            op softmax{3, op::kind::SoftMax, {masked_qk_out}, {softmax_out}, "softmax"};
            softmax.set_attr<int64_t>(op::attr::axis, -1);
            logical_tensor value_input{2, dtype, value_shape, logical_tensor::layout_type::strided};
            logical_tensor matmul_v_out{lt_id++, dtype, query_shape, logical_tensor::layout_type::strided};
            op matmul_v{4, op::kind::MatMul, {softmax_out, value_input}, {matmul_v_out}, "matmul_v"};
            g = std::make_shared<graph>(dnnl::engine::kind::gpu);
            g->add_op(matmul_qk);
            g->add_op(scale_div);
            g->add_op(mask_add);
            g->add_op(softmax);
            g->add_op(matmul_v);
            g->finalize();
            _graph = g;
            _concrete_tensors = {{0, query_input},
                                 {1, key_input},
                                 {2, value_input},
                                 {3, scale_factor},
                                 {4, attention_mask},
                                 {lt_id, matmul_v_out}};
        } else {
            logical_tensor softmax_out{lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
            op softmax{3, op::kind::SoftMax, {matmul_qk_out}, {softmax_out}, "softmax"};
            softmax.set_attr<int64_t>(op::attr::axis, -1);
            logical_tensor value_input{2, dtype, value_shape, logical_tensor::layout_type::strided};
            logical_tensor matmul_v_out{lt_id++, dtype, query_shape, logical_tensor::layout_type::strided};
            op matmul_v{4, op::kind::MatMul, {softmax_out, value_input}, {matmul_v_out}, "matmul_v"};
            g = std::make_shared<graph>(dnnl::engine::kind::gpu);
            g->add_op(matmul_qk);
            // g->add_op(scale_div);
            g->add_op(softmax);
            g->add_op(matmul_v);
            g->finalize();
            _graph = g;
            _concrete_tensors = {{0, query_input},
                                 {1, key_input},
                                 {2, value_input},
                                 //  {3, scale_factor},
                                 {lt_id, matmul_v_out}};
        }
        auto partitions = g->get_partitions();
        std::unordered_set<size_t> ids_with_any_layout;
        set_any_layout(partitions, ids_with_any_layout);
        std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;
        for (auto& p : partitions) {
            if (!p.is_supported()) {
                continue;
            }
            auto inputs = p.get_input_ports();
            auto outputs = p.get_output_ports();
            // Update input logical tensors with concrete logical tensors
            for (auto& input : inputs) {
                const auto id = input.get_id();
                // If the tensor is an output of another partition, use the cached
                // logical tensor
                if (id_to_queried_logical_tensors.find(id) != id_to_queried_logical_tensors.end()) {
                    input = id_to_queried_logical_tensors[id];
                } else {
                    // Create logical tensor with strided layout
                    input = _concrete_tensors[id];
                }
            }
            // Update output logical tensors with concrete shape and layout
            for (auto& output : outputs) {
                const auto id = output.get_id();
                output = logical_tensor{id,
                                        output.get_data_type(),
                                        DNNL_GRAPH_UNKNOWN_NDIMS,  // set output dims to unknown
                                        ids_with_any_layout.count(id) ? logical_tensor::layout_type::any
                                                                      : logical_tensor::layout_type::strided};
            }
            compiled_partition cp = p.compile(inputs, outputs, dnnl_engine);
            // Update output logical tensors with queried one
            for (auto& output : outputs) {
                const auto id = output.get_id();
                output = cp.query_logical_tensor(id);
                id_to_queried_logical_tensors[id] = output;
            }
            _compiled_partitions.push_back({p, cp, inputs, outputs});
        }
        auto last_output_id = get_last_partition_output_tensor_id();
        std::vector<size_t> queried_tensor_ids;
        for (auto& kvp : id_to_queried_logical_tensors) {
            if (kvp.first == last_output_id)  // id of output
                continue;
            queried_tensor_ids.push_back(kvp.first);
        }
        std::sort(queried_tensor_ids.begin(), queried_tensor_ids.end());
        _id_to_queried_logical_tensors = std::move(id_to_queried_logical_tensors);
        _intermediate_to_ids = std::move(queried_tensor_ids);
    }

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scaled_dot_product_attention_graph_onednn>(*this);
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param) {
        kernel_selector::sdpa_configuration config;
        auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
            if (order.empty())
                return pshape;
            auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
            for (size_t i = 0; i < order.size(); i++) {
                transposed_pshape[i] = pshape[order[i]];
            }
            return transposed_pshape;
        };
        const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
        const auto query_shape =
            transpose_pshape(impl_param.get_input_layout(0).get_partial_shape(), desc->input_q_transpose_order);
        const auto key_shape =
            transpose_pshape(impl_param.get_input_layout(1).get_partial_shape(), desc->input_k_transpose_order);
        const auto value_shape =
            transpose_pshape(impl_param.get_input_layout(2).get_partial_shape(), desc->input_v_transpose_order);
        OPENVINO_ASSERT(key_shape == value_shape, "[GPU] The shapes of key and value inputs are expected to be equal");
        for (size_t i = 0; i < query_shape.size(); ++i) {
            if (query_shape[i].is_static() && key_shape[i].is_static() && value_shape[i].is_static()) {
                if (query_shape[i].get_length() > key_shape[i].get_length()) {
                    config.broadcast_axis = desc->input_k_transpose_order[i];
                    config.group_size = query_shape[i].get_length() / key_shape[i].get_length();
                }
            }
        }
        config.head_size = query_shape[query_shape.size() - 1].get_length();
        config.heads_num = query_shape[1].get_length();
        config.kv_heads_num = key_shape[1].get_length();
        config.is_causal = desc->is_causal;
        return config;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
        auto params = get_default_params<kernel_selector::sdpa_params>(impl_param);
        auto data_inputs_num = impl_param.input_layouts.size();
        params.inputs.resize(data_inputs_num);
        for (size_t i = 0; i < data_inputs_num; i++) {
            params.inputs[i] = convert_data_tensor(impl_param.get_input_layout(i));
        }
        params.conf = get_sdpa_configuration(impl_param);
        params.input0_order = desc->input_q_transpose_order;
        params.input1_order = desc->input_k_transpose_order;
        params.input2_order = desc->input_v_transpose_order;
        params.output_order = desc->output_transpose_order;
        return params;
    }

    static std::unique_ptr<primitive_impl> create(const scaled_dot_product_attention_node& arg,
                                                  const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto sdpa_kernel_params = get_kernel_params(impl_params);
        return cldnn::make_unique<scaled_dot_product_attention_graph_onednn>(engine,
                                                                             config,
                                                                             sdpa_kernel_params,
                                                                             impl_params.input_layouts,
                                                                             impl_params.output_layouts[0]);
    }
};
// namespace detail {
// attach_scaled_dot_product_attention_graph_onednn::attach_scaled_dot_product_attention_graph_onednn() {
//     std::vector<data_types> dt = {
//         data_types::f16,
//     };
//     std::vector<format::type> fmt = {
//         format::bfyx,
//     };
//     implementation_map<scaled_dot_product_attention>::add(impl_types::onednn,
//                                                           scaled_dot_product_attention_graph_onednn::create,
//                                                           dt,
//                                                           fmt);
// }
// }  // namespace detail

std::unique_ptr<primitive_impl> ScaledDotProductAttentionImplementationManager::create_impl(
    const program_node& node,
    const kernel_impl_params& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return onednn::scaled_dot_product_attention_graph_onednn::create(
        static_cast<const scaled_dot_product_attention_node&>(node),
        params);
}

}  // namespace onednn
}  // namespace cldnn
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::scaled_dot_product_attention_graph_onednn)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_dot_product_attention)