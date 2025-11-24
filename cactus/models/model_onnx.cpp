#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace cactus {
namespace engine {

OnnxModel::OnnxModel() {
    cactus_graph_ = nullptr;
    input_node_id_ = 0;
    output_node_id_ = 0;
}

size_t OnnxModel::forward(const OnnxGraphConfig& graph_config) {
    // TODO: Implement graph construction
    return output_node_id_;
}

std::vector<float> OnnxModel::run() {
    // TODO: Implement execution
    std::vector<float> result;
    return result;
}


size_t OnnxModel::build_add(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 2) {
        throw std::runtime_error("Add operation requires exactly 2 inputs");
    }
    
    // Get input cactus node IDs from the map
    size_t input1_id = onnx_to_cactus_id_[node.inputs[0]];
    size_t input2_id = onnx_to_cactus_id_[node.inputs[1]];
    
    // TODO: For now assuming scalar operations, need to handle tensor-tensor ops
    // Placeholder: use scalar_add - actual implementation needs element-wise add
    size_t output_id = gb->add(input1_id, input2_id);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_sub(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 2) {
        throw std::runtime_error("Sub operation requires exactly 2 inputs");
    }
    
    size_t input1_id = onnx_to_cactus_id_[node.inputs[0]];
    size_t input2_id = onnx_to_cactus_id_[node.inputs[1]];
    
    size_t output_id = gb->subtract(input1_id, input2_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_mul(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 2) {
        throw std::runtime_error("Mul operation requires exactly 2 inputs");
    }
    
    size_t input1_id = onnx_to_cactus_id_[(node.inputs[0])];
    size_t input2_id = onnx_to_cactus_id_[(node.inputs[1])];
    
    size_t output_id = gb->multiply(input1_id, input2_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[(node.outputs[0])] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_div(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 2) {
        throw std::runtime_error("Div operation requires exactly 2 inputs");
    }
    
    size_t input1_id = onnx_to_cactus_id_[(node.inputs[0])];
    size_t input2_id = onnx_to_cactus_id_[(node.inputs[1])];
    
    size_t output_id = gb->divide(input1_id, input2_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[(node.outputs[0])] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_max(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 2) {
        throw std::runtime_error("Max operation requires exactly 2 inputs");
    }
    
    size_t input1_id = onnx_to_cactus_id_[(node.inputs[0])];
    size_t input2_id = onnx_to_cactus_id_[(node.inputs[1])];
    
    // Element-wise max between two tensors
    size_t output_id = gb->elem_wise_max(input1_id, input2_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[(node.outputs[0])] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_min(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 2) {
        throw std::runtime_error("Min operation requires exactly 2 inputs");
    }
    
    size_t input1_id = onnx_to_cactus_id_[(node.inputs[0])];
    size_t input2_id = onnx_to_cactus_id_[(node.inputs[1])];
    
    // Element-wise min between two tensors
    size_t output_id = gb->elem_wise_min(input1_id, input2_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[(node.outputs[0])] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_sigmoid(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 1) {
        throw std::runtime_error("Sigmoid operation requires exactly 1 input");
    }
    
    size_t input_id = onnx_to_cactus_id_[(node.inputs[0])];
    
    // Apply sigmoid activation
    size_t output_id = gb->sigmoid(input_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[(node.outputs[0])] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_sin(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 1) {
        throw std::runtime_error("Sin operation requires exactly 1 input");
    }
    
    size_t input_id = onnx_to_cactus_id_[(node.inputs[0])];
    
    // Apply element-wise sine
    size_t output_id = gb->scalar_sin(input_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[(node.outputs[0])] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_cos(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 1) {
        throw std::runtime_error("Cos operation requires exactly 1 input");
    }
    
    size_t input_id = onnx_to_cactus_id_[(node.inputs[0])];
    
    // Apply element-wise cosine
    size_t output_id = gb->scalar_cos(input_id);
    
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[(node.outputs[0])] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_batch_normalization(CactusGraph* gb, const OnnxNodeConfig& node) {
    // BatchNormalization: 5 inputs (X, scale, B, mean, var), attrs: epsilon, momentum
    // Inference mode: Y = (X - mean) / sqrt(var + epsilon) * scale + B
    // Momentum is ignored for inference mode
    if (node.inputs.size() != 5) {
        throw std::runtime_error("BatchNormalization operation requires exactly 5 inputs (X, scale, B, mean, var)");
    }
    
    // Get input cactus node IDs from the map
    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];      // X
    size_t scale_id = onnx_to_cactus_id_[node.inputs[1]];      // scale
    size_t bias_id = onnx_to_cactus_id_[node.inputs[2]];       // B
    size_t mean_id = onnx_to_cactus_id_[node.inputs[3]];       // mean
    size_t variance_id = onnx_to_cactus_id_[node.inputs[4]];   // var
    
    // Get epsilon attribute (defaults to 1e-5f in OnnxAttrConfig)
    float epsilon = node.attributes.epsilon;
    
    // Call batchnorm operation
    size_t output_id = gb->batchnorm(input_id, scale_id, bias_id, mean_id, variance_id, epsilon);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_concat(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Concat: Concatenate tensors along a specified axis
    // Inputs: list of tensors to concatenate (2 or more)
    // Attribute: axis
    if (node.inputs.size() < 2) {
        throw std::runtime_error("Concat operation requires at least 2 inputs");
    }
    
    // Get axis attribute
    int axis = static_cast<int>(node.attributes.axis);
    
    // Get first two input node IDs
    size_t result_id = onnx_to_cactus_id_[node.inputs[0]];
    
    // Chain concat operations for all inputs
    // CactusGraph::concat only supports 2 inputs at a time, so we need to chain them
    for (size_t i = 1; i < node.inputs.size(); ++i) {
        size_t next_input_id = onnx_to_cactus_id_[node.inputs[i]];
        result_id = gb->concat(result_id, next_input_id, axis);
    }
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = result_id;
    }
    
    return result_id;
}

size_t OnnxModel::build_conv(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("Conv not yet implemented");
}

size_t OnnxModel::build_conv_transpose(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("ConvTranspose not yet implemented");
}

size_t OnnxModel::build_flatten(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Flatten: Reshape tensor by keeping dims 0..axis-1, collapsing axis..r-1
    // Input: data tensor
    // Attribute: axis (default: 1)
    if (node.inputs.size() != 1) {
        throw std::runtime_error("Flatten operation requires exactly 1 input");
    }
    
    // Get input node ID
    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    
    // Get input shape
    const BufferDesc& input_buffer = gb->get_output_buffer(input_id);
    const std::vector<size_t>& input_shape = input_buffer.shape;
    
    // Get axis attribute (default: 1)
    int64_t axis = node.attributes.axis;
    if (axis == 0) {
        axis = 1;  // Default to 1 if not specified or 0
    }
    
    // Validate axis
    if (axis < 0 || axis > static_cast<int64_t>(input_shape.size())) {
        throw std::runtime_error("Flatten: axis out of range");
    }
    
    size_t axis_idx = static_cast<size_t>(axis);
    
    // Build new shape: [dim_0, dim_1, ..., dim_{axis-1}, product(dim_axis..dim_{r-1})]
    std::vector<size_t> new_shape;
    
    // Keep dimensions 0 to axis-1 as-is
    for (size_t i = 0; i < axis_idx; ++i) {
        new_shape.push_back(input_shape[i]);
    }
    
    // Collapse dimensions axis to r-1 into a single dimension
    size_t collapsed_dim = 1;
    for (size_t i = axis_idx; i < input_shape.size(); ++i) {
        collapsed_dim *= input_shape[i];
    }
    new_shape.push_back(collapsed_dim);
    
    // Apply reshape to flatten
    size_t output_id = gb->reshape(input_id, new_shape);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_gather(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("Gather not yet implemented");
}

size_t OnnxModel::build_gemm(CactusGraph* gb, const OnnxNodeConfig& node) {
    // GEMM: Y = alpha * A @ (B^T if transB) + beta * C
    // Inputs: A, B, C (C is optional)
    if (node.inputs.size() < 2) {
        throw std::runtime_error("Gemm operation requires at least 2 inputs (A, B)");
    }
    
    // Get input node IDs
    size_t input_a = onnx_to_cactus_id_[node.inputs[0]];
    size_t input_b = onnx_to_cactus_id_[node.inputs[1]];
    
    // Get attributes (defaults: alpha=1.0, beta=1.0, transB=0)
    float alpha = node.attributes.alpha != 0.0f ? node.attributes.alpha : 1.0f;
    float beta = node.attributes.beta != 0.0f ? node.attributes.beta : 1.0f;
    bool trans_b = (node.attributes.transB != 0);
    
    // Step 1: Compute A @ B (or A @ B^T if transB is set)
    size_t matmul_result = gb->matmul(input_a, input_b, trans_b);
    
    // Step 2: Multiply by alpha if not 1.0
    size_t alpha_result = matmul_result;
    if (alpha != 1.0f) {
        alpha_result = gb->scalar_multiply(matmul_result, alpha);
    }
    
    // Step 3: Add beta * C if C is provided
    size_t output_id = alpha_result;
    if (node.inputs.size() >= 3) {
        size_t input_c = onnx_to_cactus_id_[node.inputs[2]];
        
        // Multiply C by beta if not 1.0
        size_t beta_c = input_c;
        if (beta != 1.0f) {
            beta_c = gb->scalar_multiply(input_c, beta);
        }
        
        // Add alpha * (A @ B) + beta * C
        output_id = gb->add(alpha_result, beta_c);
    }
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_global_average_pool(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("GlobalAveragePool not yet implemented");
}

size_t OnnxModel::build_matmul(CactusGraph* gb, const OnnxNodeConfig& node) {
    // MatMul: Standard matrix multiplication
    if (node.inputs.size() != 2) {
        throw std::runtime_error("MatMul operation requires exactly 2 inputs");
    }
    
    // Get input node IDs
    size_t input1_id = onnx_to_cactus_id_[node.inputs[0]];
    size_t input2_id = onnx_to_cactus_id_[node.inputs[1]];
    
    // Perform matrix multiplication (no transpose)
    size_t output_id = gb->matmul(input1_id, input2_id, false);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_max_pool(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("MaxPool not yet implemented");
}

size_t OnnxModel::build_reshape(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Reshape: Y = reshape(X, shape)
    // The shape is provided via the shape attribute (list of integers)
    if (node.inputs.size() < 1) {
        throw std::runtime_error("Reshape operation requires at least 1 input (data)");
    }
    
    // Get input data node ID
    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    
    // Check for shape attribute
    if (node.attributes.kernel_shape.empty()) {
        throw std::runtime_error("Reshape operation requires 'shape' attribute");
    }
    
    const std::vector<int64_t>& shape_attr = node.attributes.kernel_shape;
    
    // Validate no zeros in shape (not supported)
    for (size_t i = 0; i < shape_attr.size(); ++i) {
        if (shape_attr[i] == 0) {
            throw std::runtime_error("Reshape with 0 dimensions is not supported");
        }
    }
    
    // Get input shape to compute total size and handle -1 dimension
    const BufferDesc& input_buffer = gb->get_output_buffer(input_id);
    const std::vector<size_t>& input_shape = input_buffer.shape;
    
    // Calculate total input size
    size_t total_input_size = 1;
    for (size_t dim : input_shape) {
        total_input_size *= dim;
    }
    
    // Convert shape attribute to size_t and find -1 dimension if present
    std::vector<size_t> new_shape;
    new_shape.reserve(shape_attr.size());
    int minus_one_idx = -1;
    size_t known_size = 1;
    
    for (size_t i = 0; i < shape_attr.size(); ++i) {
        if (shape_attr[i] == -1) {
            if (minus_one_idx != -1) {
                throw std::runtime_error("Reshape can have at most one -1 dimension");
            }
            minus_one_idx = static_cast<int>(i);
            new_shape.push_back(0);  // Placeholder, will compute later
        } else if (shape_attr[i] < 0) {
            throw std::runtime_error("Reshape shape dimensions must be positive or -1");
        } else {
            size_t dim = static_cast<size_t>(shape_attr[i]);
            new_shape.push_back(dim);
            known_size *= dim;
        }
    }
    
    // Compute -1 dimension if present
    if (minus_one_idx != -1) {
        if (total_input_size % known_size != 0) {
            throw std::runtime_error("Reshape: total input size not divisible by known dimensions");
        }
        new_shape[minus_one_idx] = total_input_size / known_size;
    } else {
        // Verify total size matches
        if (known_size != total_input_size) {
            throw std::runtime_error("Reshape: new shape total size doesn't match input size");
        }
    }
    
    // Apply reshape
    size_t output_id = gb->reshape(input_id, new_shape);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_resize(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("Resize not yet implemented");
}

size_t OnnxModel::build_slice(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Slice: ONNX provides starts, ends, steps, axes attributes
    if (node.inputs.empty()) {
        throw std::runtime_error("Slice operation requires an input tensor");
    }

    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    const auto& attrs = node.attributes;

    auto require_single_value = [&](const std::vector<int>& values, const char* name) -> int {
        if (values.empty()) {
            throw std::runtime_error(std::string("Slice requires '") + name + "' attribute");
        }
        if (values.size() != 1) {
            throw std::runtime_error(std::string("Slice static attribute '") + name + "' must contain exactly one element");
        }
        return values[0];
    };

    int start = require_single_value(attrs.slice_starts, "slice_starts");
    int end = require_single_value(attrs.slice_ends, "slice_ends");
    int axis = require_single_value(attrs.axes, "axes");

    if (start < 0) {
        throw std::runtime_error("Slice: start value must be non-negative in this implementation");
    }
    if (end <= start) {
        throw std::runtime_error("Slice: end must be greater than start for static slicing");
    }

    if (!attrs.slice_steps.empty()) {
        if (attrs.slice_steps.size() != 1) {
            throw std::runtime_error("Slice: slice_steps must contain exactly one element when provided");
        }
        if (attrs.slice_steps[0] != 1) {
            throw std::runtime_error("Slice: only step=1 is supported for static slicing");
        }
    }

    size_t length = static_cast<size_t>(end - start);
    size_t output_id = gb->slice(input_id, axis, static_cast<size_t>(start), length);

    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }

    return output_id;
}

size_t OnnxModel::build_softmax(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Softmax: Normalize along a specified axis
    // Input: data tensor
    // Attribute: axis (which dimension to normalize over)
    if (node.inputs.size() != 1) {
        throw std::runtime_error("Softmax operation requires exactly 1 input");
    }
    
    // Get input node ID
    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    
    // Get input shape to determine if we need to transpose
    const BufferDesc& input_buffer = gb->get_output_buffer(input_id);
    const std::vector<size_t>& input_shape = input_buffer.shape;
    
    // Get axis attribute (default: -1 or last dimension in ONNX)
    int axis = static_cast<int>(node.attributes.axis);
    
    // Normalize negative axis
    if (axis < 0) {
        axis += static_cast<int>(input_shape.size());
    }
    
    // Validate axis
    if (axis < 0 || axis >= static_cast<int>(input_shape.size())) {
        throw std::runtime_error("Softmax: axis out of range");
    }
    
    size_t axis_idx = static_cast<size_t>(axis);
    size_t last_idx = input_shape.size() - 1;
    
    size_t output_id;
    
    // Fast path: softmax on last dimension (no transpose needed)
    if (axis_idx == last_idx) {
        output_id = gb->softmax(input_id, axis);
    } 
    // Slow path: softmax on non-last dimension (transpose-softmax-transpose)
    else {
        // Build permutation to move axis to last position
        // Example: [0,1,2,3] with axis=1 -> [0,2,3,1]
        std::vector<size_t> perm_to_last;
        perm_to_last.reserve(input_shape.size());
        
        // Add all dimensions except the target axis
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i != axis_idx) {
                perm_to_last.push_back(i);
            }
        }
        // Add the target axis at the end
        perm_to_last.push_back(axis_idx);
        
        // Step 1: Transpose to move softmax axis to last position
        size_t transposed_input = gb->transposeN(input_id, perm_to_last);
        
        // Step 2: Apply softmax on last dimension (axis=-1)
        size_t softmax_output = gb->softmax(transposed_input, -1);
        
        // Step 3: Transpose back to original dimension order
        // Build inverse permutation
        std::vector<size_t> perm_back(input_shape.size());
        for (size_t i = 0; i < perm_to_last.size(); ++i) {
            perm_back[perm_to_last[i]] = i;
        }
        
        output_id = gb->transposeN(softmax_output, perm_back);
    }
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_split(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Split: Split input tensor into multiple output tensors based on splits attribute
    // Input: data tensor, splits (list of integers)
    // Attribute: splits (list of integers)
        
    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    const BufferDesc& input_buffer = gb->get_output_buffer(input_id);
    std::vector<int> splits = node.attributes.splits;
    int axis = static_cast<int>(node.attributes.axis);
    if (axis < 0) {
        axis += static_cast<int>(input_buffer.shape.size());
    }

    if (axis < 0 || axis >= static_cast<int>(input_buffer.shape.size())) {
        throw std::runtime_error("Split: axis out of range");
    }    

    int axis_size = input_buffer.shape[axis];

    if (splits.empty()) {
        if (axis_size % node.outputs.size() != 0) {
            throw std::runtime_error("Split: axis size not divisible by number of outputs");
        }
        splits = std::vector<int>(node.outputs.size(), axis_size / node.outputs.size());
    }

    if (!splits.empty()) {
        if (splits.size() != node.outputs.size()) {
            throw std::runtime_error("Split: splits length must match number of outputs");
        }
    }

    int sum_splits = 0;
    for (int s : splits) {
        if (s < 0) {
            throw std::runtime_error("Split: split sizes must be non-negative");
        }
        sum_splits += s;
    }
    if (sum_splits != axis_size) {
        throw std::runtime_error("Split: sum of splits must equal axis size");
    }

    

    size_t output_id;

    int index = 0;
    for (int i = 0; i < node.outputs.size(); i++) {
        output_id = gb->slice(input_id, axis, static_cast<size_t>(index), splits[i]);
        index += splits[i];
    }
    return output_id;
}

size_t OnnxModel::build_transpose(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Transpose: Generic permutation of axes
    // Input: data tensor
    // Attribute: perm (permutation of axes)
    if (node.inputs.size() != 1) {
        throw std::runtime_error("Transpose operation requires exactly 1 input");
    }
    
    // Get input node ID
    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    
    // Get permutation from attributes
    const std::vector<int64_t>& perm = node.attributes.perm;
    
    if (perm.empty()) {
        throw std::runtime_error("Transpose operation requires 'perm' attribute");
    }
    
    // Convert int64_t permutation to size_t
    std::vector<size_t> permutation;
    permutation.reserve(perm.size());
    for (int64_t p : perm) {
        if (p < 0) {
            throw std::runtime_error("Transpose permutation cannot contain negative values");
        }
        permutation.push_back(static_cast<size_t>(p));
    }
    
    // Apply transpose with the given permutation
    size_t output_id = gb->transposeN(input_id, permutation);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_unsqueeze(CactusGraph* gb, const OnnxNodeConfig& node) {
    // Unsqueeze: Add dimensions of size 1 at specified axes
    // Inputs: data (tensor), axes (1D int64 tensor or attribute)
    // In ONNX opset 13+, axes is an input; in earlier versions it's an attribute
    if (node.inputs.size() < 1) {
        throw std::runtime_error("Unsqueeze operation requires at least 1 input");
    }
    
    // Get input node ID
    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    
    // Get input shape
    const BufferDesc& input_buffer = gb->get_output_buffer(input_id);
    const std::vector<size_t>& input_shape = input_buffer.shape;
    
    // Get axes to unsqueeze
    // For now, assume axes are provided via attributes (we can extend to handle input later)
    // The axes should be in node.attributes - need to check which field
    // Based on ONNX spec, axes would typically be stored as a separate attribute
    // For this implementation, we'll use the perm field as a placeholder for axes
    // TODO: May need to add explicit 'axes' field to OnnxAttrConfig
    const std::vector<int>& axes = node.attributes.axes;
    
    if (axes.empty()) {
        throw std::runtime_error("Unsqueeze operation requires 'axes' attribute or input");
    }
    
    // Normalize axes to positive values and sort them
    std::vector<int> normalized_axes;
    int output_rank = static_cast<int>(input_shape.size()) + static_cast<int>(axes.size());

    for (int axis : axes) {
        int norm_axis = axis;
        if (norm_axis < 0) {
            norm_axis += output_rank;
        }
        if (norm_axis < 0 || norm_axis >= output_rank) {
            throw std::runtime_error("Unsqueeze: axis out of range");
        }
        normalized_axes.push_back(norm_axis);
    }
    
    // Sort axes to process in order
    std::sort(normalized_axes.begin(), normalized_axes.end());
    
    // Build new shape by inserting 1s at specified positions
    std::vector<size_t> new_shape;
    size_t input_idx = 0;
    size_t axes_idx = 0;
    
    for (int64_t i = 0; i < output_rank; ++i) {
        if (axes_idx < normalized_axes.size() && i == normalized_axes[axes_idx]) {
            // Insert dimension of size 1
            new_shape.push_back(1);
            axes_idx++;
        } else {
            // Copy from input shape
            if (input_idx >= input_shape.size()) {
                throw std::runtime_error("Unsqueeze: internal error building shape");
            }
            new_shape.push_back(input_shape[input_idx]);
            input_idx++;
        }
    }
    
    // Apply reshape to add the new dimensions
    size_t output_id = gb->reshape(input_id, new_shape);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_input(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("Input node not yet implemented");
}

size_t OnnxModel::build_weight(CactusGraph* gb, const OnnxNodeConfig& node) {
    throw std::runtime_error("Weight node not yet implemented");
}

}
}
