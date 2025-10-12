#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <set>

namespace cactus {
namespace engine {

SmolVLMModel::SmolVLMModel() : SmolModel() {}

SmolVLMModel::SmolVLMModel(const VLMConfig& cfg) : SmolModel(cfg) {
    weight_nodes_.vision_layers.resize(cfg.num_layers);
}

}
}