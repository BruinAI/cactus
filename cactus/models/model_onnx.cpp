#include "model.h"
#include "../graph/graph.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
namespace cactus {
namespace engine {

namespace {

std::vector<unsigned char> convert_to_rgb(
    const unsigned char* img_data, int width, int height, int channels) {
    std::vector<unsigned char> rgb_data(width * height * 3);

    if (channels == 1) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i];
            rgb_data[i * 3 + 1] = img_data[i];
            rgb_data[i * 3 + 2] = img_data[i];
        }
    } else if (channels == 4) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i * 4 + 0];
            rgb_data[i * 3 + 1] = img_data[i * 4 + 1];
            rgb_data[i * 3 + 2] = img_data[i * 4 + 2];
        }
    } else if (channels == 2) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i * 2 + 0];
            rgb_data[i * 3 + 1] = img_data[i * 2 + 0];
            rgb_data[i * 3 + 2] = img_data[i * 2 + 0];
        }
    } else {
        throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
    }

    return rgb_data;
}

std::vector<float> resize_image(
    const unsigned char* img_data, int src_width, int src_height,
    int dst_width, int dst_height, int channels) {
    const size_t src_elements = static_cast<size_t>(src_width) * src_height * channels;
    std::vector<float> src_float(src_elements);
    for (size_t idx = 0; idx < src_elements; ++idx) {
        src_float[idx] = static_cast<float>(img_data[idx]);
    }

    std::vector<float> resized_data(static_cast<size_t>(dst_width) * dst_height * channels);

    stbir_pixel_layout layout = (channels == 1) ? STBIR_1CHANNEL :
                                (channels == 3) ? STBIR_RGB : STBIR_RGBA;

    float* result = stbir_resize_float_linear(
        src_float.data(), src_width, src_height, 0,
        resized_data.data(), dst_width, dst_height, 0,
        layout
    );

    if (!result) {
        throw std::runtime_error("Failed to resize image");
    }

    return resized_data;
}

} // namespace

static const char* get_onnx_op_name(OnnxModel::OnnxOpType op) {
    switch (op) {
        case OnnxModel::OnnxOpType::INPUT: return "INPUT";
        case OnnxModel::OnnxOpType::WEIGHT: return "WEIGHT";
        case OnnxModel::OnnxOpType::ADD: return "ADD";
        case OnnxModel::OnnxOpType::BATCH_NORMALIZATION: return "BATCH_NORM";
        case OnnxModel::OnnxOpType::CONCAT: return "CONCAT";
        case OnnxModel::OnnxOpType::CONV: return "CONV";
        case OnnxModel::OnnxOpType::CONV_TRANSPOSE: return "CONV_TRANSPOSE";
        case OnnxModel::OnnxOpType::COS: return "COS";
        case OnnxModel::OnnxOpType::DIV: return "DIV";
        case OnnxModel::OnnxOpType::FLATTEN: return "FLATTEN";
        case OnnxModel::OnnxOpType::GATHER: return "GATHER";
        case OnnxModel::OnnxOpType::GEMM: return "GEMM";
        case OnnxModel::OnnxOpType::GLOBAL_AVERAGE_POOL: return "GLOBAL_AVG_POOL";
        case OnnxModel::OnnxOpType::MATMUL: return "MATMUL";
        case OnnxModel::OnnxOpType::MAX: return "MAX";
        case OnnxModel::OnnxOpType::MAX_POOL: return "MAX_POOL";
        case OnnxModel::OnnxOpType::MIN: return "MIN";
        case OnnxModel::OnnxOpType::MUL: return "MUL";
        case OnnxModel::OnnxOpType::RESHAPE: return "RESHAPE";
        case OnnxModel::OnnxOpType::RESIZE: return "RESIZE";
        case OnnxModel::OnnxOpType::SIGMOID: return "SIGMOID";
        case OnnxModel::OnnxOpType::SIN: return "SIN";
        case OnnxModel::OnnxOpType::SLICE: return "SLICE";
        case OnnxModel::OnnxOpType::SOFTMAX: return "SOFTMAX";
        case OnnxModel::OnnxOpType::SPLIT: return "SPLIT";
        case OnnxModel::OnnxOpType::SUB: return "SUB";
        case OnnxModel::OnnxOpType::TRANSPOSE: return "TRANSPOSE";
        case OnnxModel::OnnxOpType::UNSQUEEZE: return "UNSQUEEZE";
    }
    return "UNKNOWN";
}

static std::string make_debug_node_name(const OnnxModel::OnnxNodeConfig& node) {
    return std::string(get_onnx_op_name(node.op_type)) + "#" + std::to_string(node.onnx_node_id);
}

OnnxModel::OnnxModel()
    : cactus_graph_(new CactusGraph()),
      input_node_id_(0),
      output_node_id_(0) {
}

OnnxModel::OnnxModel(const std::string& ir_path, const std::string& default_input_path)
    : OnnxModel() {
    default_input_path_ = default_input_path;
    graph_config_ = load_graph_config_from_blob(ir_path);
    build_graph(graph_config_);
}

OnnxModel::~OnnxModel() {
    delete cactus_graph_;
    cactus_graph_ = nullptr;
}

size_t OnnxModel::build_graph(const OnnxGraphConfig& graph_config) {
    graph_config_ = graph_config;
    for (const auto& node : graph_config.nodes) {
        size_t node_output_id = 0;
        switch (node.op_type) {
            case OnnxOpType::INPUT:
                node_output_id = build_input(cactus_graph_, node);
                break;
            case OnnxOpType::WEIGHT:
                node_output_id = build_weight(cactus_graph_, node);
                break;
            case OnnxOpType::ADD:
                node_output_id = build_add(cactus_graph_, node);
                break;
            case OnnxOpType::BATCH_NORMALIZATION:
                node_output_id = build_batch_normalization(cactus_graph_, node);
                break;
            case OnnxOpType::CONCAT:
                node_output_id = build_concat(cactus_graph_, node);
                break;
            case OnnxOpType::CONV_TRANSPOSE:
                node_output_id = build_conv_transpose(cactus_graph_, node);
                break;
            case OnnxOpType::CONV:
                node_output_id = build_conv(cactus_graph_, node);
                break;
            case OnnxOpType::COS:
                node_output_id = build_cos(cactus_graph_, node);
                break;
            case OnnxOpType::DIV:
                node_output_id = build_div(cactus_graph_, node);
                break;
            case OnnxOpType::SIN:
                node_output_id = build_sin(cactus_graph_, node);
                break;
            case OnnxOpType::SLICE:
                node_output_id = build_slice(cactus_graph_, node);
                break;
            case OnnxOpType::SOFTMAX:
                node_output_id = build_softmax(cactus_graph_, node);
                break;
            case OnnxOpType::FLATTEN:
                node_output_id = build_flatten(cactus_graph_, node);
                break;
            case OnnxOpType::GATHER:
                node_output_id = build_gather(cactus_graph_, node);
                break;
            case OnnxOpType::GEMM:
                node_output_id = build_gemm(cactus_graph_, node);
                break;
            case OnnxOpType::GLOBAL_AVERAGE_POOL:
                node_output_id = build_global_average_pool(cactus_graph_, node);
                break;
            case OnnxOpType::MATMUL:
                node_output_id = build_matmul(cactus_graph_, node);
                break;
            case OnnxOpType::MAX:
                node_output_id = build_max(cactus_graph_, node);
                break;
            case OnnxOpType::MAX_POOL:
                node_output_id = build_max_pool(cactus_graph_, node);
                break;
            case OnnxOpType::MIN:
                node_output_id = build_min(cactus_graph_, node);
                break;
            case OnnxOpType::MUL:
                node_output_id = build_mul(cactus_graph_, node);
                break;
            case OnnxOpType::RESHAPE:
                node_output_id = build_reshape(cactus_graph_, node);
                break;
            case OnnxOpType::RESIZE:
                node_output_id = build_resize(cactus_graph_, node);
                break;
            case OnnxOpType::SIGMOID:
                node_output_id = build_sigmoid(cactus_graph_, node);
                break;
            case OnnxOpType::SPLIT:
                node_output_id = build_split(cactus_graph_, node);
                break;
            case OnnxOpType::SUB:
                node_output_id = build_sub(cactus_graph_, node);
                break;
            case OnnxOpType::TRANSPOSE:
                node_output_id = build_transpose(cactus_graph_, node);
                break;
            case OnnxOpType::UNSQUEEZE:
                node_output_id = build_unsqueeze(cactus_graph_, node);
                break;
            default:
                throw std::runtime_error("Unsupported operation: " + std::to_string(static_cast<int>(node.op_type)));
        }

        // if (!node.outputs.empty()) {
        //     if (node_output_id == 0) {
        //         auto it = onnx_to_cactus_id_.find(node.outputs[0]);
        //         if (it != onnx_to_cactus_id_.end()) {
        //             node_output_id = it->second;
        //         }
        //     }

        //     if (node_output_id != 0 && cactus_graph_) {
        //         cactus_graph_->capture_debug_node(
        //             static_cast<uint32_t>(node.onnx_node_id),
        //             make_debug_node_name(node),
        //             node_output_id);
        //     }
        // }
    }
    return output_node_id_;
}
std::vector<float> OnnxModel::forward(const OnnxGraphConfig& graph_config){
    cactus_graph_->execute(profile_path_);
    
    auto it = onnx_to_cactus_id_.find(graph_config.output_node_id);
    if (it == onnx_to_cactus_id_.end()) {
        throw std::runtime_error("Output node ID " + std::to_string(graph_config.output_node_id) + 
                                 " not found in onnx_to_cactus_id_ map");
    }
    const BufferDesc& output_buffer = cactus_graph_->get_output_buffer(it->second);
    
    // TODO: Re-enable output file writing when needed
    // if (!output_path_.empty()) {
    //     write_output_to_file(output_buffer, output_path_);
    // }
    
    auto return_data = std::vector<float>(output_buffer.data_as<float>(), output_buffer.data_as<float>() + output_buffer.total_size);
    return return_data;
}

std::vector<float> OnnxModel::run() {
    if (graph_config_.nodes.empty()) {
        throw std::runtime_error("Graph configuration missing");
    }
    
    // Reset graph and rebuild for each run to ensure clean state
    cactus_graph_->soft_reset();
    onnx_to_cactus_id_.clear();
    build_graph(graph_config_);
    
    return forward(graph_config_);
}

void OnnxModel::set_default_input_path(const std::string& path) {
    default_input_path_ = path;
}

void OnnxModel::set_output_path(const std::string& path) {
    output_path_ = path;
}

void OnnxModel::set_profile_path(const std::string& path) {
    profile_path_ = path;
}

void OnnxModel::write_output_to_file(const BufferDesc& buffer, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    // Write header:
    // 1. ndim (uint32)
    // 2. shape dims (uint64 each)
    // 3. precision (uint32): 0=INT8, 1=FP16, 2=FP32
    // 4. byte_size (uint64)
    // 5. scale (float32) if precision==INT8
    // 6. raw data bytes

    uint32_t ndim = static_cast<uint32_t>(buffer.shape.size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    for (size_t dim : buffer.shape) {
        uint64_t d = static_cast<uint64_t>(dim);
        file.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }

    uint32_t prec_val = static_cast<uint32_t>(buffer.precision);
    file.write(reinterpret_cast<const char*>(&prec_val), sizeof(prec_val));

    uint64_t byte_size = static_cast<uint64_t>(buffer.byte_size);
    file.write(reinterpret_cast<const char*>(&byte_size), sizeof(byte_size));

    if (buffer.precision == Precision::INT8) {
        float scale = buffer.quantization_scale;
        file.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
    }

    file.write(reinterpret_cast<const char*>(buffer.get_data()), buffer.byte_size);

    std::cout << "Wrote output to " << path << " (shape: [";
    for (size_t i = 0; i < buffer.shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << buffer.shape[i];
    }
    std::cout << "], " << buffer.total_size << " elements)" << std::endl;
}

size_t OnnxModel::build_add(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.size() != 2) {
        throw std::runtime_error("Add operation requires exactly 2 inputs");
    }
    
    // Get input cactus node IDs from the map
    size_t input1_id = onnx_to_cactus_id_[node.inputs[0]];
    size_t input2_id = onnx_to_cactus_id_[node.inputs[1]];
    
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
    // Conv: Y = conv(X, W) + B (optional bias)
    // Inputs: X (input), W (weights), B (optional bias)
    if (node.inputs.size() < 2) {
        throw std::runtime_error("Conv requires at least 2 inputs (X, W)");
    }

    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    size_t weight_id = onnx_to_cactus_id_[node.inputs[1]];
    size_t bias_id = 0;
    if (node.inputs.size() > 2) {
        bias_id = onnx_to_cactus_id_[node.inputs[2]];
    }

    const auto& input_buffer = gb->get_output_buffer(input_id);
    const auto& weight_buffer = gb->get_output_buffer(weight_id);
    const auto& attrs = node.attributes;

    // Validate 4D tensors (2D conv with NCHW)
    if (input_buffer.shape.size() != 4) {
        throw std::runtime_error("Conv requires a 4D input tensor (NCHW)");
    }
    if (weight_buffer.shape.size() != 4) {
        throw std::runtime_error("Conv requires a 4D weight tensor [C_out, C_in/groups, kH, kW]");
    }

    // Validate dilations == [1, 1]
    if (!attrs.dilations.empty()) {
        if (attrs.dilations.size() != 2) {
            throw std::runtime_error("Conv dilations must be length 2");
        }
        if (attrs.dilations[0] != 1 || attrs.dilations[1] != 1) {
            throw std::runtime_error("Conv dilation must be 1 (no atrous convolution supported)");
        }
    }

    // Get kernel shape from weight tensor (shape is [C_out, C_in/groups, kH, kW])
    size_t kernel_h = weight_buffer.shape[2];
    size_t kernel_w = weight_buffer.shape[3];
    if (kernel_h == 0 || kernel_w == 0) {
        throw std::runtime_error("Conv kernel dimensions must be > 0");
    }

    // Get strides (must be equal)
    size_t stride = 1;
    if (!attrs.strides.empty()) {
        if (attrs.strides.size() != 2) {
            throw std::runtime_error("Conv strides must be length 2");
        }
        if (attrs.strides[0] != attrs.strides[1]) {
            throw std::runtime_error("Conv strides must be equal across axes");
        }
        stride = static_cast<size_t>(attrs.strides[0]);
        if (stride == 0) {
            throw std::runtime_error("Conv stride must be > 0");
        }
    }

    // Get groups
    size_t groups = static_cast<size_t>(attrs.group);
    if (groups == 0) groups = 1;

    // Resolve padding (must be uniform)
    size_t pad = 0;

    if (attrs.auto_pad == "VALID") {
        // VALID means no padding
        pad = 0;
    } else if (attrs.auto_pad.empty() || attrs.auto_pad == "NOTSET") {
        // Use explicit pads attribute
        if (!attrs.pads.empty()) {
            if (attrs.pads.size() != 4) {
                throw std::runtime_error("Conv pads must have 4 values [top, left, bottom, right]");
            }
            int64_t pad_top = attrs.pads[0];
            int64_t pad_left = attrs.pads[1];
            int64_t pad_bottom = attrs.pads[2];
            int64_t pad_right = attrs.pads[3];

            if (pad_top != pad_bottom || pad_left != pad_right || pad_top != pad_left) {
                throw std::runtime_error("Conv pads must be equal across all sides");
            }
            if (pad_top < 0) {
                throw std::runtime_error("Conv pads must be non-negative");
            }
            pad = static_cast<size_t>(pad_top);
        }
    } else {
        throw std::runtime_error("Conv auto_pad must be 'VALID', 'NOTSET', or empty");
    }

    // Validate groups
    size_t C_in = input_buffer.shape[1];
    size_t C_out = weight_buffer.shape[0];
    if (C_in % groups != 0) {
        throw std::runtime_error("Conv: C_in must be divisible by groups");
    }
    if (C_out % groups != 0) {
        throw std::runtime_error("Conv: C_out must be divisible by groups");
    }

    size_t output_id = build_conv2d_gemm(gb, input_id, weight_id, bias_id,
                                     kernel_h, kernel_w,
                                     stride,
                                     pad,
                                     groups);

    // auto& output_buffer = gb->get_output_buffer(output_id);

    // auto output_shape = output_buffer.shape;

    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_conv2d_gemm(CactusGraph* gb, size_t input, size_t weight, size_t bias,
                                size_t kernel_h, size_t kernel_w,
                                size_t stride,
                                size_t pad,
                                size_t groups) {
    const auto& input_buffer = gb->get_output_buffer(input);
    const auto& weight_buffer = gb->get_output_buffer(weight);
    
    if (input_buffer.shape.size() != 4) {
        throw std::runtime_error("conv2d_gemm expects a 4D input tensor [N, C, H, W]");
    }
    if (weight_buffer.shape.size() != 4) {
        throw std::runtime_error("conv2d_gemm expects a 4D weight tensor [C_out, C_in/groups, kH, kW]");
    }
    
    size_t N = input_buffer.shape[0];
    size_t C_in = input_buffer.shape[1];
    size_t H = input_buffer.shape[2];
    size_t W = input_buffer.shape[3];
    size_t C_out = weight_buffer.shape[0];
    
    if (groups != 1) {
        // For grouped convolutions, fall back to direct conv2d
        return gb->conv2d(input, weight, bias, kernel_h, kernel_w, stride, pad, groups);
    }
    
    size_t H_out = (H + 2 * pad - kernel_h) / stride + 1;
    size_t W_out = (W + 2 * pad - kernel_w) / stride + 1;
    size_t patch_size = C_in * kernel_h * kernel_w;
    
    // Step 1: im2col - extract patches
    // Input: [N, C_in, H, W] -> [N, H_out * W_out, C_in * kernel_h * kernel_w]
    size_t col_id = gb->im2col(input, kernel_h, kernel_w, stride, stride, pad, pad);
    
    // Step 2: Reshape weights from [C_out, C_in, kH, kW] to [C_out, C_in * kH * kW]
    size_t weight_reshaped = gb->reshape(weight, {C_out, patch_size});
    
    // Step 3: Matrix multiply: [N, H_out * W_out, patch_size] @ [patch_size, C_out] = [N, H_out * W_out, C_out]
    // We need col @ weight.T, which is matmul_nd with pretransposed_rhs=true
    size_t matmul_result = gb->matmul_nd(col_id, weight_reshaped, true);
    
    // Step 4: Add bias if present (broadcast over spatial dimensions)
    size_t result = matmul_result;
    if (bias != 0) {
        result = gb->add(matmul_result, bias);
    }
    
    // Step 5: Reshape from [N, H_out * W_out, C_out] to [N, C_out, H_out, W_out]
    // First reshape to [N, H_out, W_out, C_out]
    size_t reshaped = gb->reshape(result, {N, H_out, W_out, C_out});
    
    // Then transpose to [N, C_out, H_out, W_out] using permutation [0, 3, 1, 2]
    size_t output = gb->transposeN(reshaped, {0, 3, 1, 2});

    return output;
}

size_t OnnxModel::build_conv_transpose(CactusGraph* gb, const OnnxNodeConfig& node) {
    // ConvTranspose: Y = conv_transpose(X, W) + B (optional bias)
    // Inputs: X (input), W (weights), B (optional bias)
    // Weight shape: [C_in, C_out_per_group, kH, kW] (note: reversed from Conv)
    if (node.inputs.size() < 2) {
        throw std::runtime_error("ConvTranspose requires at least 2 inputs (X, W)");
    }

    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    size_t weight_id = onnx_to_cactus_id_[node.inputs[1]];
    size_t bias_id = 0;
    if (node.inputs.size() > 2) {
        bias_id = onnx_to_cactus_id_[node.inputs[2]];
    }

    const auto& input_buffer = gb->get_output_buffer(input_id);
    const auto& weight_buffer = gb->get_output_buffer(weight_id);
    const auto& attrs = node.attributes;

    // Validate 4D tensors (2D conv transpose with NCHW)
    if (input_buffer.shape.size() != 4) {
        throw std::runtime_error("ConvTranspose requires a 4D input tensor (NCHW)");
    }
    if (weight_buffer.shape.size() != 4) {
        throw std::runtime_error("ConvTranspose requires a 4D weight tensor [C_in, C_out/groups, kH, kW]");
    }

    // Validate dilations == [1, 1]
    if (!attrs.dilations.empty()) {
        if (attrs.dilations.size() != 2) {
            throw std::runtime_error("ConvTranspose dilations must be length 2");
        }
        if (attrs.dilations[0] != 1 || attrs.dilations[1] != 1) {
            throw std::runtime_error("ConvTranspose dilation must be 1 (no atrous convolution supported)");
        }
    }

    // Get kernel shape from weight tensor (shape is [C_in, C_out/groups, kH, kW])
    size_t kernel_h = weight_buffer.shape[2];
    size_t kernel_w = weight_buffer.shape[3];
    if (kernel_h == 0 || kernel_w == 0) {
        throw std::runtime_error("ConvTranspose kernel dimensions must be > 0");
    }

    // Get strides (must be equal)
    size_t stride = 1;
    if (!attrs.strides.empty()) {
        if (attrs.strides.size() != 2) {
            throw std::runtime_error("ConvTranspose strides must be length 2");
        }
        if (attrs.strides[0] != attrs.strides[1]) {
            throw std::runtime_error("ConvTranspose strides must be equal across axes");
        }
        stride = static_cast<size_t>(attrs.strides[0]);
        if (stride == 0) {
            throw std::runtime_error("ConvTranspose stride must be > 0");
        }
    }

    // Get groups
    size_t groups = static_cast<size_t>(attrs.group);
    if (groups == 0) groups = 1;

    // Resolve padding (must be uniform)
    size_t pad = 0;

    if (attrs.auto_pad == "VALID") {
        // VALID means no padding
        pad = 0;
    } else if (attrs.auto_pad.empty() || attrs.auto_pad == "NOTSET") {
        // Use explicit pads attribute
        if (!attrs.pads.empty()) {
            if (attrs.pads.size() != 4) {
                throw std::runtime_error("ConvTranspose pads must have 4 values [top, left, bottom, right]");
            }
            int64_t pad_top = attrs.pads[0];
            int64_t pad_left = attrs.pads[1];
            int64_t pad_bottom = attrs.pads[2];
            int64_t pad_right = attrs.pads[3];

            if (pad_top != pad_bottom || pad_left != pad_right || pad_top != pad_left) {
                throw std::runtime_error("ConvTranspose pads must be equal across all sides");
            }
            if (pad_top < 0) {
                throw std::runtime_error("ConvTranspose pads must be non-negative");
            }
            pad = static_cast<size_t>(pad_top);
        }
    } else {
        throw std::runtime_error("ConvTranspose auto_pad must be 'VALID', 'NOTSET', or empty");
    }

    // Validate groups
    size_t C_in = input_buffer.shape[1];
    size_t C_out = weight_buffer.shape[1] * groups;
    if (C_in % groups != 0) {
        throw std::runtime_error("ConvTranspose: C_in must be divisible by groups");
    }
    if (C_out % groups != 0) {
        throw std::runtime_error("ConvTranspose: C_out must be divisible by groups");
    }

    size_t output_id = gb->conv_transpose2d(input_id, weight_id, bias_id,
                                             kernel_h, kernel_w,
                                             stride,
                                             pad,
                                             groups);

    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }

    return output_id;
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
    if (node.inputs.size() < 2) {
        throw std::runtime_error("Gather requires data and indices inputs");
    }

    size_t data_id = onnx_to_cactus_id_[node.inputs[0]];
    size_t indices_id = onnx_to_cactus_id_[node.inputs[1]];

    const BufferDesc& data_buf = gb->get_output_buffer(data_id);
    if (data_buf.shape.empty()) {
        throw std::runtime_error("Gather requires non-scalar data tensor");
    }

    int axis = static_cast<int>(node.attributes.axis);
    int rank = static_cast<int>(data_buf.shape.size());
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error("Gather axis out of range");
    }

    size_t output_id = 0;

    if (axis == 0) {
        output_id = gb->gather(data_id, indices_id);
    } else {
        std::vector<size_t> perm;
        perm.reserve(rank);
        perm.push_back(static_cast<size_t>(axis));
        for (size_t d = 0; d < data_buf.shape.size(); ++d) {
            if (static_cast<int>(d) == axis) {
                continue;
            }
            perm.push_back(d);
        }

        size_t transposed_data = gb->transposeN(data_id, perm);
        size_t gathered = gb->gather(transposed_data, indices_id);

        const BufferDesc& indices_buf = gb->get_output_buffer(indices_id);
        size_t indices_rank = indices_buf.shape.size();
        size_t gathered_rank = indices_rank + data_buf.shape.size() - 1;

        std::vector<size_t> perm_after;
        perm_after.reserve(gathered_rank);
        size_t axis_block_start = indices_rank;
        size_t axis_block_end = axis_block_start + static_cast<size_t>(axis);
        for (size_t i = axis_block_start; i < axis_block_end; ++i) {
            perm_after.push_back(i);
        }
        for (size_t i = 0; i < indices_rank; ++i) {
            perm_after.push_back(i);
        }
        for (size_t i = axis_block_end; i < gathered_rank; ++i) {
            perm_after.push_back(i);
        }

        output_id = gb->transposeN(gathered, perm_after);
    }

    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }

    return output_id;
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
    if (node.inputs.size() != 1) {
        throw std::runtime_error("GlobalAveragePool requires exactly 1 input");
    }

    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    const auto& input_buffer = gb->get_output_buffer(input_id);

    if (input_buffer.shape.size() != 4) {
        throw std::runtime_error("GlobalAveragePool expects a 4D tensor in NCHW layout");
    }

    size_t output_id = gb->global_avg_pool(input_id);

    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }

    return output_id;
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
    size_t output_id = gb->matmul_nd(input1_id, input2_id, false);
    
    // Store output mapping
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    
    return output_id;
}

size_t OnnxModel::build_max_pool(CactusGraph* gb, const OnnxNodeConfig& node) {
    if (node.inputs.empty()) {
        throw std::runtime_error("MaxPool requires at least one input");
    }

    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    const auto& input_buffer = gb->get_output_buffer(input_id);
    const auto& attrs = node.attributes;

    if (input_buffer.shape.size() != 4) {
        throw std::runtime_error("MaxPool requires a 4D input tensor (NCHW)");
    }

    if (attrs.auto_pad != "" && attrs.auto_pad != "NOTSET") {
        throw std::runtime_error("MaxPool auto_pad must be empty or NOTSET");
    }

    if (attrs.kernel_shape.size() != 2) {
        throw std::runtime_error("MaxPool requires a 2D kernel_shape");
    }

    size_t kernel_h = static_cast<size_t>(attrs.kernel_shape[0]);
    size_t kernel_w = static_cast<size_t>(attrs.kernel_shape[1]);
    if (kernel_h == 0 || kernel_w == 0) {
        throw std::runtime_error("MaxPool kernel dimensions must be > 0");
    }

    size_t stride = 1;
    if (!attrs.strides.empty()) {
        if (attrs.strides.size() != 2) throw std::runtime_error("MaxPool strides must be length 2");
        if (attrs.strides[0] != attrs.strides[1]) {
            throw std::runtime_error("MaxPool strides must be equal across axes");
        }
        stride = static_cast<size_t>(attrs.strides[0]);
        if (stride == 0) throw std::runtime_error("MaxPool stride must be > 0");
    }

    size_t dilation = 1;
    if (!attrs.dilations.empty()) {
        if (attrs.dilations.size() != 2) throw std::runtime_error("MaxPool dilations must be length 2");
        if (attrs.dilations[0] != attrs.dilations[1]) {
            throw std::runtime_error("MaxPool dilations must be equal across axes");
        }
        dilation = static_cast<size_t>(attrs.dilations[0]);
        if (dilation == 0) throw std::runtime_error("MaxPool dilation must be > 0");
    }

    size_t pad = 0;
    if (!attrs.pads.empty()) {
        if (attrs.pads.size() != 4) {
            throw std::runtime_error("MaxPool pads must have 4 values");
        }
        int64_t pad_top = attrs.pads[0];
        int64_t pad_left = attrs.pads[1];
        int64_t pad_bottom = attrs.pads[2];
        int64_t pad_right = attrs.pads[3];

        if (pad_top != pad_bottom || pad_left != pad_right || pad_top != pad_left) {
            throw std::runtime_error("MaxPool pads must be equal across all sides");
        }
        if (pad_top < 0) {
            throw std::runtime_error("MaxPool pads must be non-negative");
        }
        pad = static_cast<size_t>(pad_top);
    }

    size_t output_id = gb->maxpool(input_id, kernel_h, kernel_w, stride, pad, dilation);
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }

    return output_id;
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
    if (node.attributes.shape.empty()) {
        throw std::runtime_error("Reshape operation requires 'shape' attribute");
    }
    
    const std::vector<int64_t>& shape_attr = node.attributes.shape;
    
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
    if (node.inputs.empty()) {
        throw std::runtime_error("Resize requires at least one input");
    }

    size_t input_id = onnx_to_cactus_id_[node.inputs[0]];
    const auto& input_buffer = gb->get_output_buffer(input_id);
    const auto& attrs = node.attributes;

    if (attrs.roi) {
        throw std::runtime_error("Resize ROI input must be empty for this implementation");
    }
    if (attrs.antialias) {
        throw std::runtime_error("Resize antialias must be disabled");
    }
    if (!attrs.mode.empty() && attrs.mode != "nearest") {
        throw std::runtime_error("Resize mode must be 'nearest'");
    }
    if (!attrs.coordinate_transformation_mode.empty() && attrs.coordinate_transformation_mode != "asymmetric") {
        throw std::runtime_error("Resize coordinate_transformation_mode must be 'asymmetric'");
    }
    if (!attrs.nearest_mode.empty() && attrs.nearest_mode != "floor") {
        throw std::runtime_error("Resize nearest_mode must be 'floor'");
    }
    if (attrs.keep_aspect_ratio_policy != "" && attrs.keep_aspect_ratio_policy != "stretch") {
        throw std::runtime_error("Resize keep_aspect_ratio_policy must be 'stretch' (or unspecified)");
    }
    if (attrs.exclude_outside != 0) {
        throw std::runtime_error("Resize exclude_outside must be 0");
    }
    if (std::fabs(attrs.extrapolation_value) > 1e-6f) {
        throw std::runtime_error("Resize extrapolation_value must be 0");
    }

    const auto& scales = attrs.scales;
    if (scales.empty()) {
        throw std::runtime_error("Resize requires static scales attribute");
    }

    size_t rank = input_buffer.shape.size();
    if (rank < 2) {
        throw std::runtime_error("Resize requires at least 2D input");
    }
    if (scales.size() != rank) {
        throw std::runtime_error("Resize: scales.size() must match input rank");
    }

    const float eps = 1e-6f;
    std::vector<size_t> resized_axes;
    for (size_t i = 0; i < rank; ++i) {
        if (std::fabs(scales[i] - 1.0f) > eps) {
            resized_axes.push_back(i);
        }
    }
    if (resized_axes.size() > 2) {
        throw std::runtime_error("Resize: more than two resized axes not supported");
    }
    for (size_t axis : resized_axes) {
        if (axis < rank - 2) {
            throw std::runtime_error("Resize: only last two dimensions may be resized");
        }
    }

    std::vector<size_t> out_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
        float scaled = static_cast<float>(input_buffer.shape[i]) * scales[i];
        size_t dim = static_cast<size_t>(std::floor(scaled + eps));
        out_shape[i] = dim == 0 ? 1 : dim;
    }

    if (resized_axes.empty()) {
        if (!node.outputs.empty()) {
            onnx_to_cactus_id_[node.outputs[0]] = input_id;
        }
        return input_id;
    }

    const size_t src_h = input_buffer.shape[rank - 2];
    const size_t src_w = input_buffer.shape[rank - 1];
    const size_t dst_h = out_shape[rank - 2];
    const size_t dst_w = out_shape[rank - 1];

    size_t outer_count = 1;
    for (size_t i = 0; i < rank - 2; ++i) {
        outer_count *= input_buffer.shape[i];
    }

    std::vector<size_t> collapsed_shape = {outer_count, src_h, src_w};
    size_t reshaped_input = gb->reshape(input_id, collapsed_shape);

    size_t resized_id = gb->resize_nearest_asymmetric(reshaped_input, dst_h, dst_w);

    size_t final_id = gb->reshape(resized_id, out_shape);

    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = final_id;
    }

    return final_id;
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
        std::cerr << "Split: sum of splits (" << sum_splits << ") does not equal axis size (" << axis_size << ")" << std::endl;
        for (size_t i = 0; i < splits.size(); ++i) {
            std::cerr << "  splits[" << i << "] = " << splits[i] << std::endl;
        }
        throw std::runtime_error("Split: sum of splits must equal axis size");
    }

    

    size_t output_id;

    int index = 0;
    for (int i = 0; i < node.outputs.size(); i++) {
        output_id = gb->slice(input_id, axis, static_cast<size_t>(index), splits[i]);
        index += splits[i];
        if (!node.outputs.empty() && i < node.outputs.size()) {
            onnx_to_cactus_id_[node.outputs[i]] = output_id;
        }
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
    std::string path_to_weights = node.attributes.path_to_weights; 
    if (path_to_weights.empty()) {
        path_to_weights = default_input_path_;
    }
    Precision input_precision = node.attributes.input_precision;
    if (path_to_weights.empty()) {
        throw std::runtime_error("Input node requires 'path_to_weights' attribute or default input path");
    }
    auto input_data = OnnxModel::preprocess_input(path_to_weights, node);

    std::vector<size_t> input_shape = node.attributes.input_shape;
    size_t output_id = gb->input(input_shape, input_precision);
    gb->set_input(output_id, input_data.data(), input_precision);
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    return output_id;

}

size_t OnnxModel::build_weight(CactusGraph* gb, const OnnxNodeConfig& node) {
    std::string path_to_weights = node.attributes.path_to_weights;
    size_t output_id = gb->mmap_weights(path_to_weights);
    if (!node.outputs.empty()) {
        onnx_to_cactus_id_[node.outputs[0]] = output_id;
    }
    return output_id;
}

std::vector<float> OnnxModel::preprocess_input(const std::string& path_to_weights, const OnnxNodeConfig& node) {
    using Clock = std::chrono::high_resolution_clock;
    auto total_start = Clock::now();
    
    if (path_to_weights.empty()) {
        throw std::runtime_error("Input node requires path_to_weights pointing to an image file");
    }

    const auto& shape = node.attributes.input_shape;
    if (shape.size() < 3) {
        throw std::runtime_error("Input shape must include at least C, H, W dimensions");
    }

    size_t channels = shape[shape.size() - 3];
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];

    if (channels == 0 || height == 0 || width == 0) {
        throw std::runtime_error("Input shape dimensions must be positive");
    }

    int target_height = static_cast<int>(height);
    int target_width = static_cast<int>(width);

    // Profile: Image loading
    auto load_start = Clock::now();
    int src_width = 0;
    int src_height = 0;
    int src_channels = 0;
    unsigned char* img_data = stbi_load(path_to_weights.c_str(), &src_width, &src_height, &src_channels, 0);
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + path_to_weights);
    }
    auto load_end = Clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    // Profile: RGB conversion
    auto convert_start = Clock::now();
    std::vector<unsigned char> rgb_data;
    const unsigned char* source_data = img_data;
    int active_channels = src_channels;
    if (active_channels != 3) {
        rgb_data = convert_to_rgb(img_data, src_width, src_height, active_channels);
        source_data = rgb_data.data();
        active_channels = 3;
    }
    auto convert_end = Clock::now();
    double convert_ms = std::chrono::duration<double, std::milli>(convert_end - convert_start).count();

    // Profile: Resize
    auto resize_start = Clock::now();
    auto resized = resize_image(source_data, src_width, src_height, target_width, target_height, active_channels);
    stbi_image_free(img_data);
    img_data = nullptr;
    auto resize_end = Clock::now();
    double resize_ms = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();

    if (static_cast<size_t>(active_channels) != channels) {
        throw std::runtime_error("Input shape channel count does not match loaded image channels");
    }

    // Profile: Layout conversion (HWC -> CHW) and normalization
    auto layout_start = Clock::now();
    std::vector<float> input_data(channels * height * width);
    size_t hw = height * width;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t dst_idx = c * hw + y * width + x;
                size_t src_idx = (y * width + x) * active_channels + c;
                input_data[dst_idx] = resized[src_idx] / 255.0f;
            }
        }
    }
    auto layout_end = Clock::now();
    double layout_ms = std::chrono::duration<double, std::milli>(layout_end - layout_start).count();

    auto total_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Print profiling results (can be disabled by setting env var CACTUS_QUIET_PREPROCESS=1)
    const char* quiet = std::getenv("CACTUS_QUIET_PREPROCESS");
    if (!quiet || std::string(quiet) != "1") {
        std::cout << "[Preprocess] Image: " << src_width << "x" << src_height << "x" << src_channels 
                  << " -> " << target_width << "x" << target_height << "x" << channels << std::endl;
        std::cout << "[Preprocess] Load:    " << std::fixed << std::setprecision(3) << load_ms << " ms" << std::endl;
        std::cout << "[Preprocess] Convert: " << std::fixed << std::setprecision(3) << convert_ms << " ms" << std::endl;
        std::cout << "[Preprocess] Resize:  " << std::fixed << std::setprecision(3) << resize_ms << " ms" << std::endl;
        std::cout << "[Preprocess] Layout:  " << std::fixed << std::setprecision(3) << layout_ms << " ms" << std::endl;
        std::cout << "[Preprocess] Total:   " << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;
    }

    return input_data;
}



namespace {

constexpr char kBinaryMagic[] = "CAIR";
constexpr uint8_t kBinaryVersion = 1;

enum class BinaryAttrType : uint8_t {
    Float = 0,
    Int64 = 1,
    Bool = 2,
    String = 3,
    FloatArray = 4,
    Int64Array = 5,
};

using AttrConfig = OnnxModel::OnnxAttrConfig;

template <typename T>
T read_scalar(std::ifstream& stream) {
    T value;
    if (!stream.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        throw std::runtime_error("Failed to read binary IR");
    }
    return value;
}

std::string read_binary_string(std::ifstream& stream) {
    uint32_t length = read_scalar<uint32_t>(stream);
    std::string result(length, '\0');
    if (length > 0 && !stream.read(result.data(), length)) {
        throw std::runtime_error("Failed to read binary IR string");
    }
    return result;
}

std::vector<float> read_float_array(std::ifstream& stream) {
    uint32_t count = read_scalar<uint32_t>(stream);
    std::vector<float> values(count);
    if (count > 0 && !stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float))) {
        throw std::runtime_error("Failed to read binary IR float array");
    }
    return values;
}

std::vector<int64_t> read_int64_array(std::ifstream& stream) {
    uint32_t count = read_scalar<uint32_t>(stream);
    std::vector<int64_t> values(count);
    if (count > 0 && !stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(int64_t))) {
        throw std::runtime_error("Failed to read binary IR int64 array");
    }
    return values;
}

std::vector<int> to_int(const std::vector<int64_t>& values) {
    std::vector<int> result;
    result.reserve(values.size());
    for (int64_t value : values) {
        result.push_back(static_cast<int>(value));
    }
    return result;
}

std::vector<int64_t> copy_int64(const std::vector<int64_t>& values) {
    return values;
}

std::vector<size_t> to_size_t(const std::vector<int64_t>& values) {
    std::vector<size_t> result;
    result.reserve(values.size());
    for (int64_t value : values) {
        result.push_back(static_cast<size_t>(value < 0 ? 0 : value));
    }
    return result;
}

bool apply_attr_float(AttrConfig& attrs, const std::string& key, float value) {
    if (key == "epsilon") {
        attrs.epsilon = value;
        return true;
    }
    if (key == "alpha") {
        attrs.alpha = value;
        return true;
    }
    if (key == "beta") {
        attrs.beta = value;
        return true;
    }
    if (key == "cubic_coeff_a") {
        attrs.cubic_coeff_a = value;
        return true;
    }
    if (key == "momentum") {
        attrs.momentum = value;
        return true;
    }
    if (key == "extrapolation_value") {
        attrs.extrapolation_value = value;
        return true;
    }
    return false;
}

bool apply_attr_int64(AttrConfig& attrs, const std::string& key, int64_t value) {
    if (key == "axis") {
        attrs.axis = value;
        return true;
    }
    if (key == "transB") {
        attrs.transB = value;
        return true;
    }
    if (key == "allowzero") {
        attrs.allowzero = value;
        return true;
    }
    if (key == "ceil_mode") {
        attrs.ceil_mode = value;
        return true;
    }
    if (key == "exclude_outside") {
        attrs.exclude_outside = value;
        return true;
    }
    if (key == "storage_order") {
        attrs.storage_order = value;
        return true;
    }
    if (key == "group") {
        attrs.group = value;
        return true;
    }
    if (key == "num_outputs") {
        attrs.num_outputs = value;
        return true;
    }
    if (key == "input_precision") {
        if (value >= 0 && value <= static_cast<int64_t>(Precision::FP32)) {
            attrs.input_precision = static_cast<Precision>(value);
        }
        return true;
    }
    return false;
}

bool apply_attr_bool(AttrConfig& attrs, const std::string& key, bool value) {
    if (key == "antialias") {
        attrs.antialias = value;
        return true;
    }
    if (key == "roi") {
        attrs.roi = value;
        return true;
    }
    if (key == "is_graph_output") {
        attrs.is_graph_output = value;
        return true;
    }
    return false;
}

bool apply_attr_string(AttrConfig& attrs, const std::string& key, const std::string& value) {
    if (key == "path_to_weights") {
        attrs.path_to_weights = value;
        return true;
    }
    if (key == "auto_pad") {
        attrs.auto_pad = value;
        return true;
    }
    if (key == "keep_aspect_ratio_policy") {
        attrs.keep_aspect_ratio_policy = value;
        return true;
    }
    if (key == "mode") {
        attrs.mode = value;
        return true;
    }
    if (key == "coordinate_transformation_mode") {
        attrs.coordinate_transformation_mode = value;
        return true;
    }
    if (key == "nearest_mode") {
        attrs.nearest_mode = value;
        return true;
    }
    return false;
}

bool apply_attr_int_list(AttrConfig& attrs, const std::string& key, const std::vector<int64_t>& values) {
    if (key == "input_shape") {
        attrs.input_shape = to_size_t(values);
        return true;
    }
    if (key == "axes") {
        attrs.axes = to_int(values);
        return true;
    }
    if (key == "slice_starts") {
        attrs.slice_starts = to_int(values);
        return true;
    }
    if (key == "slice_ends") {
        attrs.slice_ends = to_int(values);
        return true;
    }
    if (key == "slice_steps") {
        attrs.slice_steps = to_int(values);
        return true;
    }
    if (key == "splits") {
        attrs.splits = to_int(values);
        return true;
    }
    if (key == "perm") {
        attrs.perm = copy_int64(values);
        return true;
    }
    if (key == "pads") {
        attrs.pads = copy_int64(values);
        return true;
    }
    if (key == "kernel_shape") {
        attrs.kernel_shape = copy_int64(values);
        return true;
    }
    if (key == "shape") {
        attrs.shape = copy_int64(values);
        return true;
    }
    if (key == "strides") {
        attrs.strides = copy_int64(values);
        return true;
    }
    if (key == "dilations") {
        attrs.dilations = copy_int64(values);
        return true;
    }
    if (key == "sizes") {
        attrs.sizes = copy_int64(values);
        return true;
    }
    return false;
}

bool apply_attr_float_list(AttrConfig& attrs, const std::string& key, const std::vector<float>& values) {
    if (key == "scales") {
        attrs.scales = values;
        return true;
    }
    return false;
}

} // namespace

OnnxModel::OnnxGraphConfig OnnxModel::load_graph_config_from_blob(const std::string& ir_path) {
    std::ifstream stream(ir_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open IR file: " + ir_path);
    }

    char magic[4];
    stream.read(magic, sizeof(magic));
    if (std::string(magic, sizeof(magic)) != kBinaryMagic) {
        throw std::runtime_error("Invalid IR magic header");
    }

    uint8_t version = read_scalar<uint8_t>(stream);
    if (version != kBinaryVersion) {
        throw std::runtime_error("Unsupported IR version");
    }

    uint32_t node_count = read_scalar<uint32_t>(stream);
    read_scalar<uint32_t>(stream); // value count

    OnnxGraphConfig config;
    config.nodes.reserve(node_count);

    const uint8_t max_op = static_cast<uint8_t>(OnnxOpType::UNSQUEEZE);
    for (uint32_t idx = 0; idx < node_count; ++idx) {
        uint32_t node_id = read_scalar<uint32_t>(stream);
        uint8_t op_type_byte = read_scalar<uint8_t>(stream);
        if (op_type_byte > max_op) {
            throw std::runtime_error("Unsupported op type in IR");
        }

        OnnxNodeConfig node;
        node.onnx_node_id = static_cast<size_t>(node_id);
        node.op_type = static_cast<OnnxOpType>(op_type_byte);

        uint32_t input_count = read_scalar<uint32_t>(stream);
        node.inputs.resize(input_count);
        for (uint32_t i = 0; i < input_count; ++i) {
            node.inputs[i] = static_cast<int>(read_scalar<uint32_t>(stream));
        }

        uint32_t output_count = read_scalar<uint32_t>(stream);
        node.outputs.resize(output_count);
        for (uint32_t i = 0; i < output_count; ++i) {
            node.outputs[i] = static_cast<int>(read_scalar<uint32_t>(stream));
        }

        uint32_t attr_count = read_scalar<uint32_t>(stream);
        for (uint32_t attr_index = 0; attr_index < attr_count; ++attr_index) {
            std::string key = read_binary_string(stream);
            BinaryAttrType type = static_cast<BinaryAttrType>(read_scalar<uint8_t>(stream));
            switch (type) {
                case BinaryAttrType::Float:
                    apply_attr_float(node.attributes, key, read_scalar<float>(stream));
                    break;
                case BinaryAttrType::Int64:
                    apply_attr_int64(node.attributes, key, read_scalar<int64_t>(stream));
                    break;
                case BinaryAttrType::Bool:
                    apply_attr_bool(node.attributes, key, read_scalar<uint8_t>(stream) != 0);
                    break;
                case BinaryAttrType::String:
                    apply_attr_string(node.attributes, key, read_binary_string(stream));
                    break;
                case BinaryAttrType::FloatArray:
                    apply_attr_float_list(node.attributes, key, read_float_array(stream));
                    break;
                case BinaryAttrType::Int64Array:
                    apply_attr_int_list(node.attributes, key, read_int64_array(stream));
                    break;
                default:
                    throw std::runtime_error("Unknown attribute type in IR");
            }
        }

        config.nodes.push_back(std::move(node));
    }

    if (!config.nodes.empty()) {
        config.input_node_id = 0;
        // Find the first node marked as graph output, or fall back to last node
        config.output_node_id = -1;
        for (size_t i = 0; i < config.nodes.size(); ++i) {
            if (config.nodes[i].attributes.is_graph_output) {
                // Use the first output value of the first graph output node
                if (!config.nodes[i].outputs.empty()) {
                    config.output_node_id = config.nodes[i].outputs[0];
                }
                break;
            }
        }
        // Fall back to last node's first output if no graph output was marked
        if (config.output_node_id == -1 && !config.nodes.back().outputs.empty()) {
            config.output_node_id = config.nodes.back().outputs[0];
        }
    } else {
        config.input_node_id = -1;
        config.output_node_id = -1;
    }

    return config;
}

}
}