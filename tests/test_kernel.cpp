#include "test_utils.h"
#include <vector>
#include <cmath>
#include <iostream>

bool test_neon_add_correctness() {
    const size_t size = 16;
    std::vector<int8_t> a(size), b(size), result(size), expected(size);
    
    TestUtils::fill_random_int8(a);
    TestUtils::fill_random_int8(b);
    
    for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(a[i]) + static_cast<int>(b[i]))));
    }
    
    cactus_add_int8(a.data(), b.data(), result.data(), size);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), size);
}

bool test_neon_subtract_correctness() {
    const size_t size = 16;
    std::vector<int8_t> a(size), b(size), result(size), expected(size);
    
    TestUtils::fill_random_int8(a);
    TestUtils::fill_random_int8(b);
    
    for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(a[i]) - static_cast<int>(b[i]))));
    }
    
    cactus_subtract_int8(a.data(), b.data(), result.data(), size);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), size);
}

bool test_neon_hadamard_correctness() {
    const size_t size = 16;
    std::vector<int8_t> a(size), b(size), result(size), expected(size);
    
    TestUtils::fill_random_int8(a);
    TestUtils::fill_random_int8(b);
    
    for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(a[i]) * static_cast<int>(b[i]))));
    }
    
    cactus_multiply_int8(a.data(), b.data(), result.data(), size);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), size);
}

bool test_neon_scalar_operations_correctness() {
    const size_t size = 8;
    std::vector<int8_t> input = {1, 2, 3, 4, -1, -2, -3, -4};
    std::vector<int8_t> result(size);
    const float scalar = 2.0f;
    
    std::vector<int8_t> expected_add(size);
    for (size_t i = 0; i < size; ++i) {
        expected_add[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(input[i] + scalar))));
    }
    
    cactus_scalar_op_int8(input.data(), result.data(), size, scalar, ScalarOpType::ADD);
    
    if (!TestUtils::compare_arrays(result.data(), expected_add.data(), size)) {
        return false;
    }
    
    std::vector<int8_t> expected_mul(size);
    for (size_t i = 0; i < size; ++i) {
        expected_mul[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(input[i] * scalar))));
    }
    
    cactus_scalar_op_int8(input.data(), result.data(), size, scalar, ScalarOpType::MULTIPLY);
    
    return TestUtils::compare_arrays(result.data(), expected_mul.data(), size);
}

bool test_neon_matrix_multiply_correctness() {
    const size_t M = 4, K = 3, N = 2;
    std::vector<int8_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int8_t> b = {1, 2, 3, 4, 5, 6};
    std::vector<int8_t> b_transposed = {1, 3, 5, 2, 4, 6};
    std::vector<int8_t> result(M * N, 0);
    
    std::vector<int8_t> expected = {22, 28, 49, 64, 76, 100, 103, 127};
    
    cactus_matmul_int8(a.data(), b_transposed.data(), result.data(), M, K, N, 1.0f, 1.0f, 1.0f);
    
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(expected[i]))));
    }
    
    return TestUtils::compare_arrays(result.data(), expected.data(), M * N);
}

bool test_neon_reduction_correctness() {
    std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8};
    
    int64_t sum_result = cactus_sum_all_int8(input.data(), input.size());
    int64_t expected_sum = 36; 
    
    if (sum_result != expected_sum) {
        return false;
    }
    
    double mean_result = cactus_mean_all_int8(input.data(), input.size());
    double expected_mean = 4.5; 
    
    if (std::abs(mean_result - expected_mean) > 1e-6) {
        return false;
    }
    
    return true;
}

bool test_neon_transpose_correctness() {
    const size_t M = 3, N = 4;
    std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int8_t> result(M * N);
    std::vector<int8_t> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    
    size_t shape[] = {M, N};
    size_t perm[] = {1, 0};
    
    cactus_transpose_int8(input.data(), result.data(), shape, perm, 2, 0, M);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), M * N);
}

bool test_f16_transpose_2d_correctness() {
    // Test small 3x4 matrix
    const size_t M = 3, N = 4;
    std::vector<__fp16> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<__fp16> result(M * N);
    std::vector<__fp16> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    
    cactus_transpose_2d_f16(input.data(), result.data(), M, N, 0, M);
    
    for (size_t i = 0; i < M * N; ++i) {
        if (std::abs(static_cast<float>(result[i]) - static_cast<float>(expected[i])) > 1e-3f) {
            std::cerr << "F16 transpose 2d mismatch at " << i << ": got " << static_cast<float>(result[i]) 
                      << " expected " << static_cast<float>(expected[i]) << std::endl;
            return false;
        }
    }
    return true;
}

bool test_f16_transpose_2d_large() {
    // Test larger matrix to exercise NEON path (8x8 blocks)
    const size_t M = 16, N = 16;
    std::vector<__fp16> input(M * N);
    std::vector<__fp16> result(M * N);
    
    // Fill with test pattern
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            input[i * N + j] = static_cast<__fp16>(i * N + j);
        }
    }
    
    cactus_transpose_2d_f16(input.data(), result.data(), M, N, 0, M);
    
    // Verify transpose: result[j][i] should equal input[i][j]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __fp16 expected = input[i * N + j];
            __fp16 actual = result[j * M + i];
            if (std::abs(static_cast<float>(actual) - static_cast<float>(expected)) > 1e-3f) {
                std::cerr << "F16 transpose large mismatch at [" << j << "][" << i << "]: got " 
                          << static_cast<float>(actual) << " expected " << static_cast<float>(expected) << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool test_f16_transpose_general_correctness() {
    // Test general N-D transpose function
    const size_t M = 3, N = 4;
    std::vector<__fp16> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<__fp16> result(M * N);
    std::vector<__fp16> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    
    size_t shape[] = {M, N};
    size_t perm[] = {1, 0};
    
    cactus_transpose_f16(input.data(), result.data(), shape, perm, 2, 0, M * N);
    
    for (size_t i = 0; i < M * N; ++i) {
        if (std::abs(static_cast<float>(result[i]) - static_cast<float>(expected[i])) > 1e-3f) {
            std::cerr << "F16 transpose general mismatch at " << i << ": got " << static_cast<float>(result[i]) 
                      << " expected " << static_cast<float>(expected[i]) << std::endl;
            return false;
        }
    }
    return true;
}

bool test_f16_matmul_correctness() {
    // Test: A[2x3] @ B[3x2] = C[2x2]
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[1, 2], [3, 4], [5, 6]]
    // B_transposed = [[1, 3, 5], [2, 4, 6]]
    // C = [[22, 28], [49, 64]]
    const size_t M = 2, K = 3, N = 2;
    std::vector<__fp16> a = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> b_transposed = {1, 3, 5, 2, 4, 6};  // transposed for matmul
    std::vector<__fp16> result(M * N, 0);
    std::vector<float> expected = {22.0f, 28.0f, 49.0f, 64.0f};
    
    cactus_matmul_f16(a.data(), b_transposed.data(), result.data(), M, K, N);
    
    for (size_t i = 0; i < M * N; ++i) {
        float actual = static_cast<float>(result[i]);
        if (std::abs(actual - expected[i]) > 1e-2f) {
            std::cerr << "F16 matmul mismatch at " << i << ": got " << actual 
                      << " expected " << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool test_f16_matmul_with_transpose() {
    // Test the combined transpose + matmul path used in MATMUL_ND
    // A[2x3] @ B[3x2] = C[2x2]
    const size_t M = 2, K = 3, N = 2;
    std::vector<__fp16> a = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> b = {1, 2, 3, 4, 5, 6};  // NOT transposed
    std::vector<__fp16> b_transposed(K * N);
    std::vector<__fp16> result(M * N, 0);
    std::vector<float> expected = {22.0f, 28.0f, 49.0f, 64.0f};
    
    // Transpose B from [K x N] to [N x K]
    cactus_transpose_2d_f16(b.data(), b_transposed.data(), K, N, 0, K);
    
    // Now b_transposed should be [N x K] layout
    cactus_matmul_f16(a.data(), b_transposed.data(), result.data(), M, K, N);
    
    for (size_t i = 0; i < M * N; ++i) {
        float actual = static_cast<float>(result[i]);
        if (std::abs(actual - expected[i]) > 1e-2f) {
            std::cerr << "F16 matmul with transpose mismatch at " << i << ": got " << actual 
                      << " expected " << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool test_f32_matmul_correctness() {
    // Same test as FP16 for comparison
    const size_t M = 2, K = 3, N = 2;
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_transposed = {1, 3, 5, 2, 4, 6};
    std::vector<float> result(M * N, 0);
    std::vector<float> expected = {22.0f, 28.0f, 49.0f, 64.0f};
    
    cactus_matmul_f32(a.data(), b_transposed.data(), result.data(), M, K, N);
    
    for (size_t i = 0; i < M * N; ++i) {
        if (std::abs(result[i] - expected[i]) > 1e-4f) {
            std::cerr << "F32 matmul mismatch at " << i << ": got " << result[i] 
                      << " expected " << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool test_neon_softmax_correctness() {
    const size_t batch_size = 1, seq_len = 4, vocab_size = 3;
    std::vector<float> input = {1.0f, 2.0f, 3.0f,
                               2.0f, 3.0f, 4.0f,
                               3.0f, 4.0f, 5.0f,
                               4.0f, 5.0f, 6.0f};
    std::vector<float> result(input.size());
    
    cactus_softmax_f32(input.data(), result.data(), batch_size, seq_len, vocab_size);
    
    for (size_t i = 0; i < seq_len; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            row_sum += result[i * vocab_size + j];
        }
        if (std::abs(row_sum - 1.0f) > 1e-5f) {
            return false;
        }
    }
    
    return true;
}


bool test_neon_rope_correctness() {
    const size_t batch_size = 1, seq_len = 2, num_heads = 1, head_dim = 4;
    const size_t start_pos = 0;
    const float theta = 10000.0f;
    const size_t total_elements = batch_size * seq_len * num_heads * head_dim;
    
    std::vector<float> input(total_elements);
    std::vector<float> result(total_elements);
    
    TestUtils::fill_random_float(input);
    
    cactus_rope_f32(input.data(), result.data(), 
                   batch_size, seq_len, num_heads, head_dim, start_pos, theta);
    
    bool different_from_input = false;
    for (size_t i = 0; i < total_elements; ++i) {
        if (std::abs(result[i] - input[i]) > 1e-6f) {
            different_from_input = true;
            break;
        }
    }
    
    return different_from_input;
}

bool test_neon_attention_correctness() {
    const size_t batch_size = 1, seq_len = 2, num_heads = 1, head_dim = 4;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t total_elements = batch_size * seq_len * num_heads * head_dim;
    
    std::vector<int8_t> queries(total_elements);
    std::vector<int8_t> keys(total_elements);
    std::vector<int8_t> values(total_elements);
    std::vector<int8_t> result(total_elements);
    
    TestUtils::fill_random_int8(queries);
    TestUtils::fill_random_int8(keys);
    TestUtils::fill_random_int8(values);
    
    cactus_attention_int8(queries.data(), keys.data(), values.data(), result.data(),
                         batch_size, seq_len, seq_len, num_heads, num_heads, head_dim, scale, nullptr,
                         1.0f, 1.0f, 1.0f, 1.0f);
    
    bool has_non_zero = false;
    for (size_t i = 0; i < total_elements; ++i) {
        if (result[i] != 0) {
            has_non_zero = true;
            break;
        }
    }
    
    return has_non_zero;
}


int main() {
    TestUtils::TestRunner runner("Kernel Backend Tests");
    
    runner.run_test("Kernel Add Correctness", test_neon_add_correctness());
    runner.run_test("Kernel Subtract Correctness", test_neon_subtract_correctness());
    runner.run_test("Kernel Multiply Correctness", test_neon_hadamard_correctness());
    runner.run_test("Kernel Scalar Operations Correctness", test_neon_scalar_operations_correctness());
    runner.run_test("Kernel Matrix Multiply Correctness", test_neon_matrix_multiply_correctness());
    runner.run_test("Kernel Reduction Correctness", test_neon_reduction_correctness());
    runner.run_test("Kernel Transpose INT8 Correctness", test_neon_transpose_correctness());
    runner.run_test("Kernel Transpose F16 2D Small", test_f16_transpose_2d_correctness());
    runner.run_test("Kernel Transpose F16 2D Large (NEON)", test_f16_transpose_2d_large());
    runner.run_test("Kernel Transpose F16 General N-D", test_f16_transpose_general_correctness());
    runner.run_test("Kernel MatMul F16 Correctness", test_f16_matmul_correctness());
    runner.run_test("Kernel MatMul F16 With Transpose", test_f16_matmul_with_transpose());
    runner.run_test("Kernel MatMul F32 Correctness", test_f32_matmul_correctness());
    runner.run_test("Kernel Softmax Correctness", test_neon_softmax_correctness());
    runner.run_test("Kernel RoPE Correctness", test_neon_rope_correctness());
    runner.run_test("Kernel Attention Correctness", test_neon_attention_correctness());
    
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}