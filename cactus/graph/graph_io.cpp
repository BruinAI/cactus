#include "graph.h"
#include "../kernel/kernel.h"
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace {
    constexpr uint32_t CACTUS_MAGIC = 0x54434143;  // "CACT" in little-endian
    constexpr uint32_t TENSOR_FORMAT_VERSION = 1;
    constexpr uint32_t FLAG_HAS_SCALES = 1 << 0;
    constexpr size_t HEADER_SIZE = 80;

    inline size_t align_offset(size_t offset, size_t alignment) {
        size_t remainder = offset % alignment;
        if (remainder == 0) return offset;
        return offset + (alignment - remainder);
    }
}


size_t CactusGraph::mmap_embeddings(const std::string& filename) {
    auto mapped_file = std::make_unique<GraphFile::MappedFile>(filename);

    const auto& shape = mapped_file->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Memory-mapped embeddings must be 2D [vocab_size, embedding_dim]");
    }

    Precision precision = mapped_file->effective_precision();

    size_t node_id = input(shape, precision);
    set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);

    if (precision == Precision::INT8 && mapped_file->group_size() > 0) {
        set_grouped_scales(node_id, mapped_file->group_size(), mapped_file->num_groups(),
                          const_cast<void*>(mapped_file->scales_data()));
    }

    size_t file_idx = mapped_files_.size();
    mapped_files_.push_back(std::move(mapped_file));
    node_to_mapped_file_[node_id] = file_idx;
    return node_id;
}

size_t CactusGraph::mmap_weights(const std::string& filename) {
    auto it = weight_cache_.find(filename);
    if (it != weight_cache_.end()) {
        return it->second;
    }

    auto mapped_file = std::make_unique<GraphFile::MappedFile>(filename);

    const auto& shape = mapped_file->shape();
    Precision precision = mapped_file->effective_precision();

    size_t node_id = input(shape, precision);
    set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);

    if (precision == Precision::INT8 && mapped_file->group_size() > 0) {
        set_grouped_scales(node_id, mapped_file->group_size(), mapped_file->num_groups(),
                          const_cast<void*>(mapped_file->scales_data()));
    }

    size_t file_idx = mapped_files_.size();
    mapped_files_.push_back(std::move(mapped_file));
    node_to_mapped_file_[node_id] = file_idx;
    weight_cache_[filename] = node_id;
    return node_id;
}

void CactusGraph::release_weight_pages(size_t node_id) {
    auto it = node_to_mapped_file_.find(node_id);
    if (it != node_to_mapped_file_.end() && it->second < mapped_files_.size()) {
        mapped_files_[it->second]->release_pages();
    }
}

void CactusGraph::prefetch_weight_pages(size_t node_id) {
    auto it = node_to_mapped_file_.find(node_id);
    if (it != node_to_mapped_file_.end() && it->second < mapped_files_.size()) {
        mapped_files_[it->second]->prefetch_pages();
    }
}

void CactusGraph::release_all_weight_pages() {
    for (auto& mf : mapped_files_) {
        if (mf) mf->release_pages();
    }
}

size_t CactusGraph::load_weights(const std::string& filename) {
    auto it = weight_cache_.find(filename);
    if (it != weight_cache_.end()) {
        return it->second;
    }

    auto loaded = GraphFile::load_into_graph(*this, filename);
    weight_cache_[filename] = loaded.node_id;
    return loaded.node_id;
}

void CactusGraph::set_grouped_scales(size_t node_id, size_t group_size, size_t num_groups, void* scales_ptr) {
    auto it = node_index_map_.find(node_id);
    if (it != node_index_map_.end()) {
        nodes_[it->second]->output_buffer.set_grouped_scales(group_size, num_groups, scales_ptr);
    }
}

size_t CactusGraph::embedding(const std::string& filename, size_t indices) {
    auto mapped_file = std::make_unique<GraphFile::MappedFile>(filename);

    const auto& shape = mapped_file->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Embedding file must contain 2D tensor [vocab_size, hidden_dim]");
    }

    Precision precision = mapped_file->effective_precision();
    size_t embeddings_node = input(shape, precision);
    set_external_input(embeddings_node, const_cast<void*>(mapped_file->data()), precision);

    if (precision == Precision::INT8 && mapped_file->group_size() > 0) {
        set_grouped_scales(embeddings_node, mapped_file->group_size(), mapped_file->num_groups(),
                          const_cast<void*>(mapped_file->scales_data()));
    }

    mapped_files_.push_back(std::move(mapped_file));

    const auto& idx_shape = get_output_buffer(indices).shape;
    std::vector<size_t> output_shape = idx_shape;
    output_shape.push_back(shape[1]);

    OpParams params;
    params.output_precision = (precision == Precision::INT8) ? Precision::FP16 : precision;

    return add_node(OpType::EMBEDDING, {embeddings_node, indices}, output_shape, params);
}


namespace GraphFile {

void save_node(CactusGraph& graph, size_t node_id, const std::string& filename) {
    graph.execute();
    void* data = graph.get_output(node_id);

    const auto& buffer = graph.get_output_buffer(node_id);
    const auto& shape = buffer.shape;
    Precision precision = buffer.precision;

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    size_t total_elements = 1;
    for (size_t dim : shape) {
        total_elements *= dim;
    }

    size_t element_size = PrecisionTraits::size_of(precision);
    size_t byte_size = total_elements * element_size;

    bool has_scales = (precision == Precision::INT8 && buffer.is_grouped_int8() && buffer.scales_data);
    size_t N = shape.size() >= 1 ? shape[0] : 1;
    size_t scales_bytes = has_scales ? (N * buffer.num_groups * sizeof(__fp16)) : 0;

    uint32_t ndim = static_cast<uint32_t>(shape.size());
    uint32_t flags = has_scales ? FLAG_HAS_SCALES : 0;
    uint32_t alignment = 32;

    uint32_t magic = CACTUS_MAGIC;
    uint32_t version = TENSOR_FORMAT_VERSION;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    file.write(reinterpret_cast<const char*>(&alignment), sizeof(alignment));
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    for (uint32_t i = 0; i < 4; i++) {
        uint64_t dim_val = (i < shape.size()) ? static_cast<uint64_t>(shape[i]) : 0;
        file.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
    }

    uint32_t prec_val = static_cast<uint32_t>(precision);
    file.write(reinterpret_cast<const char*>(&prec_val), sizeof(prec_val));

    uint64_t data_bytes = static_cast<uint64_t>(byte_size);
    uint64_t scales_bytes_val = static_cast<uint64_t>(scales_bytes);
    file.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
    file.write(reinterpret_cast<const char*>(&scales_bytes_val), sizeof(scales_bytes_val));

    uint32_t group_size = has_scales ? static_cast<uint32_t>(buffer.group_size) : 0;
    uint32_t num_groups = has_scales ? static_cast<uint32_t>(buffer.num_groups) : 0;
    file.write(reinterpret_cast<const char*>(&group_size), sizeof(group_size));
    file.write(reinterpret_cast<const char*>(&num_groups), sizeof(num_groups));

    size_t header_end = 80;
    size_t aligned_header = align_offset(header_end, alignment);
    size_t header_padding = aligned_header - header_end;
    for (size_t i = 0; i < header_padding; i++) {
        char zero = 0;
        file.write(&zero, 1);
    }

    if (has_scales) {
        file.write(static_cast<const char*>(buffer.scales_data), scales_bytes);

        size_t scales_end = aligned_header + scales_bytes;
        size_t data_start = align_offset(scales_end, alignment);
        size_t scales_padding = data_start - scales_end;
        for (size_t i = 0; i < scales_padding; i++) {
            char zero = 0;
            file.write(&zero, 1);
        }
    }

    file.write(static_cast<const char*>(data), byte_size);

    if (!file) {
        throw std::runtime_error("Error writing node data to file: " + filename);
    }
}

LoadedNode load_into_graph(CactusGraph& graph, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    char header[HEADER_SIZE];
    file.read(header, HEADER_SIZE);
    if (!file) {
        throw std::runtime_error("Error reading file header: " + filename);
    }

    const char* ptr = header;
    size_t offset = 0;

    uint32_t magic = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    if (magic != CACTUS_MAGIC) {
        throw std::runtime_error("Invalid tensor file: missing CACT magic number");
    }

    offset += sizeof(uint32_t);
    offset += sizeof(uint32_t);

    uint32_t alignment = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    if (alignment == 0) alignment = 1;

    uint32_t ndim = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    std::vector<size_t> shape;
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t dim_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
        offset += sizeof(uint64_t);
        if (i < ndim && dim_val > 0) {
            shape.push_back(static_cast<size_t>(dim_val));
        }
    }

    uint32_t prec_val = *reinterpret_cast<const uint32_t*>(ptr + offset);
    Precision precision = static_cast<Precision>(prec_val);
    offset += sizeof(uint32_t);

    size_t byte_size = *reinterpret_cast<const uint64_t*>(ptr + offset);
    offset += sizeof(uint64_t);

    size_t scales_bytes = *reinterpret_cast<const uint64_t*>(ptr + offset);
    offset += sizeof(uint64_t);

    size_t group_size = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    size_t num_groups = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    size_t aligned_header = align_offset(HEADER_SIZE, alignment);
    size_t padding = aligned_header - HEADER_SIZE;
    if (padding > 0) {
        file.seekg(padding, std::ios::cur);
    }

    std::vector<char> scales_buffer;
    if (scales_bytes > 0) {
        scales_buffer.resize(scales_bytes);
        file.read(scales_buffer.data(), scales_bytes);

        size_t scales_end = aligned_header + scales_bytes;
        size_t data_start = align_offset(scales_end, alignment);
        size_t scales_padding = data_start - scales_end;
        if (scales_padding > 0) {
            file.seekg(scales_padding, std::ios::cur);
        }
    }

    std::vector<char> buffer(byte_size);
    file.read(buffer.data(), byte_size);

    if (!file || file.gcount() != static_cast<std::streamsize>(byte_size)) {
        throw std::runtime_error("Error reading node data: " + filename);
    }

    size_t node_id = graph.input(shape, precision);
    graph.set_input(node_id, buffer.data(), precision);

    if (scales_bytes > 0 && group_size > 0 && num_groups > 0) {
        auto& node_buffer = graph.nodes_[graph.node_index_map_.at(node_id)]->output_buffer;
        node_buffer.owned_scales = std::make_unique<char[]>(scales_buffer.size());
        std::memcpy(node_buffer.owned_scales.get(), scales_buffer.data(), scales_buffer.size());
        node_buffer.set_grouped_scales(group_size, num_groups, node_buffer.owned_scales.get());
    }

    return {node_id, shape, precision, byte_size};
}

MappedFile mmap_load(const std::string& filename) {
    return MappedFile(filename);
}

// MappedFile implementation

MappedFile::MappedFile(const std::string& filename)
    : fd_(-1), mapped_data_(nullptr), file_size_(0), data_offset_(0) {
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Cannot open file for mapping: " + filename);
    }

    struct stat st;
    if (fstat(fd_, &st) == -1) {
        close(fd_);
        throw std::runtime_error("Cannot get file size: " + filename);
    }
    file_size_ = st.st_size;

    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Cannot map file: " + filename);
    }

    close(fd_);
    fd_ = -1;

    parse_header();
    apply_madvise_hints();
}

MappedFile::~MappedFile() {
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
        madvise(mapped_data_, file_size_, MADV_DONTNEED);
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd_(other.fd_), mapped_data_(other.mapped_data_), file_size_(other.file_size_),
      data_offset_(other.data_offset_), shape_(std::move(other.shape_)),
      precision_(other.precision_), byte_size_(other.byte_size_),
      group_size_(other.group_size_), num_groups_(other.num_groups_),
      scales_offset_(other.scales_offset_), scales_bytes_(other.scales_bytes_),
      version_(other.version_), alignment_(other.alignment_),
      is_int4_(other.is_int4_),
      unpacked_int4_data_(std::move(other.unpacked_int4_data_)) {
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.is_int4_ = false;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
            munmap(mapped_data_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }

        fd_ = other.fd_;
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        data_offset_ = other.data_offset_;
        shape_ = std::move(other.shape_);
        precision_ = other.precision_;
        byte_size_ = other.byte_size_;
        group_size_ = other.group_size_;
        num_groups_ = other.num_groups_;
        scales_offset_ = other.scales_offset_;
        scales_bytes_ = other.scales_bytes_;
        version_ = other.version_;
        alignment_ = other.alignment_;
        is_int4_ = other.is_int4_;
        unpacked_int4_data_ = std::move(other.unpacked_int4_data_);

        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
        other.is_int4_ = false;
    }
    return *this;
}

const std::vector<size_t>& MappedFile::shape() const {
    return shape_;
}

Precision MappedFile::precision() const {
    return precision_;
}

size_t MappedFile::byte_size() const {
    return byte_size_;
}

const void* MappedFile::scales_data() const {
    return static_cast<const char*>(mapped_data_) + scales_offset_;
}

const void* MappedFile::raw_packed_data() const {
    return static_cast<const char*>(mapped_data_) + data_offset_;
}

void MappedFile::unpack_int4_if_needed() const {
    if (!is_int4_ || unpacked_int4_data_) {
        return;
    }

    size_t unpacked_count = byte_size_ * 2;
    unpacked_int4_data_ = std::make_unique<int8_t[]>(unpacked_count);

    const uint8_t* packed = reinterpret_cast<const uint8_t*>(
        static_cast<const char*>(mapped_data_) + data_offset_);

    cactus_unpack_int4_to_int8(packed, unpacked_int4_data_.get(), unpacked_count);
}

void* MappedFile::data() {
    if (is_int4_) {
        unpack_int4_if_needed();
        return unpacked_int4_data_.get();
    }
    return static_cast<char*>(mapped_data_) + data_offset_;
}

const void* MappedFile::data() const {
    if (is_int4_) {
        unpack_int4_if_needed();
        return unpacked_int4_data_.get();
    }
    return static_cast<const char*>(mapped_data_) + data_offset_;
}

template<typename T>
const T* MappedFile::typed_data() const {
    return static_cast<const T*>(data());
}

LoadedNode MappedFile::load_into_graph(CactusGraph& graph) const {
    Precision eff_prec = effective_precision();

    size_t node_id = graph.input(shape_, eff_prec);

    if (is_int4_) {
        auto& node_buffer = graph.nodes_[graph.node_index_map_.at(node_id)]->output_buffer;
        node_buffer.set_external(const_cast<void*>(raw_packed_data()));
        node_buffer.set_packed_int4(raw_packed_data(), byte_size_);
        node_buffer.byte_size = byte_size_;

        if (group_size_ > 0) {
            graph.set_grouped_scales(node_id, group_size_, num_groups_,
                                     const_cast<void*>(scales_data()));
        }
    } else {
        graph.set_external_input(node_id, const_cast<void*>(data()), eff_prec);

        if ((eff_prec == Precision::INT8) && group_size_ > 0) {
            graph.set_grouped_scales(node_id, group_size_, num_groups_,
                                     const_cast<void*>(scales_data()));
        }
    }

    return {node_id, shape_, eff_prec, byte_size_};
}

void MappedFile::parse_header() {
    if (file_size_ < HEADER_SIZE) {
        throw std::runtime_error("File too small: insufficient data for header");
    }

    const char* ptr = static_cast<const char*>(mapped_data_);
    size_t offset = 0;

    uint32_t magic = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    if (magic != CACTUS_MAGIC) {
        throw std::runtime_error("Invalid tensor file: missing CACT magic number");
    }

    version_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    uint32_t flags = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    (void)flags;

    alignment_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    if (alignment_ == 0) alignment_ = 1;

    uint32_t ndim = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    shape_.clear();
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t dim_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
        offset += sizeof(uint64_t);
        if (i < ndim && dim_val > 0) {
            shape_.push_back(static_cast<size_t>(dim_val));
        }
    }

    uint32_t prec_val = *reinterpret_cast<const uint32_t*>(ptr + offset);
    precision_ = static_cast<Precision>(prec_val);
    offset += sizeof(uint32_t);

    is_int4_ = (precision_ == Precision::INT4);

    byte_size_ = *reinterpret_cast<const uint64_t*>(ptr + offset);
    offset += sizeof(uint64_t);

    scales_bytes_ = *reinterpret_cast<const uint64_t*>(ptr + offset);
    offset += sizeof(uint64_t);

    group_size_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    num_groups_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    size_t aligned_header = align_offset(HEADER_SIZE, alignment_);

    if (scales_bytes_ > 0) {
        scales_offset_ = aligned_header;
        size_t scales_end = scales_offset_ + scales_bytes_;
        data_offset_ = align_offset(scales_end, alignment_);
    } else {
        scales_offset_ = 0;
        data_offset_ = aligned_header;
    }

    if (data_offset_ + byte_size_ > file_size_) {
        throw std::runtime_error("File corrupted: data extends beyond file size (v2)");
    }
}

void MappedFile::apply_madvise_hints() {
    if (scales_bytes_ > 0 && scales_offset_ > 0) {
        madvise(static_cast<char*>(mapped_data_) + scales_offset_, scales_bytes_, MADV_WILLNEED);
    }

    madvise(static_cast<char*>(mapped_data_) + data_offset_, byte_size_, MADV_SEQUENTIAL);

    if (byte_size_ > 1024 * 1024) {
        madvise(static_cast<char*>(mapped_data_) + data_offset_, byte_size_, MADV_WILLNEED);
    }
}

void MappedFile::release_pages() {
    if (mapped_data_ == nullptr || mapped_data_ == MAP_FAILED) return;

    if (scales_bytes_ > 0 && scales_offset_ > 0) {
        madvise(static_cast<char*>(mapped_data_) + scales_offset_, scales_bytes_, MADV_DONTNEED);
    }
    madvise(static_cast<char*>(mapped_data_) + data_offset_, byte_size_, MADV_DONTNEED);

    if (is_int4_ && unpacked_int4_data_) {
        unpacked_int4_data_.reset();
    }
}

void MappedFile::prefetch_pages() {
    if (mapped_data_ == nullptr || mapped_data_ == MAP_FAILED) return;

    if (scales_bytes_ > 0 && scales_offset_ > 0) {
        madvise(static_cast<char*>(mapped_data_) + scales_offset_, scales_bytes_, MADV_WILLNEED);
    }
    madvise(static_cast<char*>(mapped_data_) + data_offset_, byte_size_, MADV_WILLNEED);
}

template const int8_t* MappedFile::typed_data<int8_t>() const;
template const float* MappedFile::typed_data<float>() const;
template const uint16_t* MappedFile::typed_data<uint16_t>() const;
template const uint8_t* MappedFile::typed_data<uint8_t>() const;

} // namespace GraphFile
