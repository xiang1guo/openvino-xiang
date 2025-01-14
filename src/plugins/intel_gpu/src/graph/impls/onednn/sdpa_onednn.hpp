// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <oneapi/dnnl/dnnl_graph.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include <ostream>

#include "impls/onednn/utils.hpp"
#include "impls/registry/implementation_manager.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_graph_base.hpp"
#include "scaled_dot_product_attention_inst.h"

namespace cldnn {
namespace onednn {

struct ScaledDotProductAttentionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::sdpa")
    ScaledDotProductAttentionImplementationManager(shape_types shape_type)
        : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<scaled_dot_product_attention>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;

        const auto& sdpa_node = node.as<scaled_dot_product_attention>();
        const auto& in_layout = sdpa_node.get_input_layout(0);
        const auto& out_layout = sdpa_node.get_output_layout(0);
        auto in0_dt = in_layout.data_type;
        auto out_dt = out_layout.data_type;

        if (!everyone_is(data_types::f16, in_dt, wei_dt))
            return false;

        if (!everyone_is(format::bfyx, in_layout.format, out_layout.format) &&
            !everyone_is(format::any, in_layout.format, out_layout.format))
            return false;

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<scaled_dot_product_attention>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;
        }
        out_fmts[0] = format::get_default_format(out_rank);

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
