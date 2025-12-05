#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../cactus/models/model.h"

using namespace cactus::engine;
namespace fs = std::filesystem;

struct BenchmarkConfig {
    int warmup_runs = 10;
    int benchmark_runs = 50;
    std::string ir_path;
    std::string input_path;
    std::string profile_path;
    std::string precision = "fp16";  // "fp16", "fp32", or "int8"
};

struct BenchmarkResults {
    std::vector<double> times_ms;
    double total_ms;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double median_ms;
    double p95_ms;
    double p99_ms;
};

BenchmarkResults compute_stats(const std::vector<double>& times) {
    BenchmarkResults results;
    results.times_ms = times;
    
    if (times.empty()) {
        return results;
    }
    
    // Total
    results.total_ms = std::accumulate(times.begin(), times.end(), 0.0);
    
    // Mean
    results.mean_ms = results.total_ms / times.size();
    
    // Std deviation
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - results.mean_ms) * (t - results.mean_ms);
    }
    results.std_ms = std::sqrt(sq_sum / times.size());
    
    // Min/Max
    results.min_ms = *std::min_element(times.begin(), times.end());
    results.max_ms = *std::max_element(times.begin(), times.end());
    
    // Percentiles (need sorted copy)
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    
    size_t n = sorted_times.size();
    results.median_ms = sorted_times[n / 2];
    results.p95_ms = sorted_times[static_cast<size_t>(n * 0.95)];
    results.p99_ms = sorted_times[static_cast<size_t>(n * 0.99)];
    
    return results;
}

void print_results(const BenchmarkResults& results, int num_runs) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "         BENCHMARK RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "Runs:        " << num_runs << "\n";
    std::cout << "Total time:  " << results.total_ms << " ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Mean:        " << results.mean_ms << " ms\n";
    std::cout << "Std Dev:     " << results.std_ms << " ms\n";
    std::cout << "Min:         " << results.min_ms << " ms\n";
    std::cout << "Max:         " << results.max_ms << " ms\n";
    std::cout << "Median:      " << results.median_ms << " ms\n";
    std::cout << "P95:         " << results.p95_ms << " ms\n";
    std::cout << "P99:         " << results.p99_ms << " ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Throughput:  " << (1000.0 / results.mean_ms) << " inferences/sec\n";
    std::cout << "========================================\n";
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <ir_path> <input_path> [precision] [warmup_runs] [benchmark_runs] [profile_path]\n";
    std::cerr << "\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  ir_path         Path to binary IR file (graph.bin)\n";
    std::cerr << "  input_path      Path to input image\n";
    std::cerr << "  precision       Input precision: 'fp16', 'fp32', or 'int8' (default: fp16)\n";
    std::cerr << "  warmup_runs     Number of warmup runs (default: 10)\n";
    std::cerr << "  benchmark_runs  Number of benchmark runs (default: 50)\n";
    std::cerr << "  profile_path    Path to write per-op profile (optional, enables profiling on last run)\n";
}

int main(int argc, char** argv) {
    BenchmarkConfig config;
    
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    config.ir_path = argv[1];
    config.input_path = argv[2];
    
    if (argc > 3) {
        config.precision = argv[3];
        // Validate precision
        if (config.precision != "fp16" && config.precision != "fp32" && config.precision != "int8") {
            std::cerr << "Error: precision must be 'fp16', 'fp32', or 'int8', got: " << config.precision << "\n";
            return 1;
        }
    }
    if (argc > 4) {
        config.warmup_runs = std::atoi(argv[4]);
    }
    if (argc > 5) {
        config.benchmark_runs = std::atoi(argv[5]);
    }
    if (argc > 6) {
        config.profile_path = argv[6];
    }
    
    // Validate paths
    if (!fs::exists(config.ir_path)) {
        std::cerr << "Error: IR file not found: " << config.ir_path << "\n";
        return 1;
    }
    if (!fs::exists(config.input_path)) {
        std::cerr << "Error: Input file not found: " << config.input_path << "\n";
        return 1;
    }
    
    // Convert precision string to Precision enum
    // Note: INT8 weights use FP16 inputs (same preprocessing as FP16)
    Precision input_precision;
    if (config.precision == "fp32") {
        input_precision = Precision::FP32;
    } else {
        input_precision = Precision::FP16;
    }
    
    std::cout << "========================================\n";
    std::cout << "       ONNX Model Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "IR Path:         " << config.ir_path << "\n";
    std::cout << "Input Path:      " << config.input_path << "\n";
    std::cout << "Input Precision: " << config.precision << "\n";
    std::cout << "Warmup Runs:     " << config.warmup_runs << "\n";
    std::cout << "Benchmark Runs:  " << config.benchmark_runs << "\n";
    if (!config.profile_path.empty()) {
        std::cout << "Profile Path:    " << config.profile_path << "\n";
    }
    std::cout << "========================================\n\n";
    
    try {
        // Load model once
        std::cout << "Loading model..." << std::flush;
        auto load_start = std::chrono::high_resolution_clock::now();
        OnnxModel model(config.ir_path, config.input_path, input_precision);
        auto load_end = std::chrono::high_resolution_clock::now();
        double load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
        std::cout << " done (" << std::fixed << std::setprecision(1) << load_time_ms << " ms)\n\n";
        
        // Warmup runs
        std::cout << "Warmup (" << config.warmup_runs << " runs)..." << std::flush;
        for (int i = 0; i < config.warmup_runs; ++i) {
            auto output = model.run();
            if (output.empty()) {
                std::cerr << "\nError: Model produced no output on warmup run " << i << "\n";
                return 1;
            }
        }
        std::cout << " done\n\n";
        
        // Benchmark runs
        std::cout << "Benchmarking (" << config.benchmark_runs << " runs)...\n";
        std::vector<double> times;
        times.reserve(config.benchmark_runs);
        
        for (int i = 0; i < config.benchmark_runs; ++i) {
            // Enable profiling on the last run if profile_path is set
            bool is_last_run = (i == config.benchmark_runs - 1);
            if (is_last_run && !config.profile_path.empty()) {
                model.set_profile_path(config.profile_path);
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto output = model.run();
            auto end = std::chrono::high_resolution_clock::now();
            
            // Disable profiling after the profiled run
            if (is_last_run && !config.profile_path.empty()) {
                model.set_profile_path("");
            }
            
            if (output.empty()) {
                std::cerr << "Error: Model produced no output on benchmark run " << i << "\n";
                return 1;
            }
            
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(elapsed_ms);
            
            // Progress indicator every 10 runs
            if ((i + 1) % 10 == 0 || i == config.benchmark_runs - 1) {
                std::cout << "  Run " << std::setw(3) << (i + 1) << "/" << config.benchmark_runs 
                          << ": " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
            }
        }
        
        // Compute and print statistics
        BenchmarkResults results = compute_stats(times);
        print_results(results, config.benchmark_runs);
        
        return 0;
        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

