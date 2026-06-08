// SPDX-License-Identifier: 0BSD
// candidate.cu — only the marked blocks may be edited. The locked driver
// measures the cuBLAS reference and your custom kernel back-to-back over
// a fixed problem size and prints one structured RESULT line at the end.

#include <chrono>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <typeinfo>
// BEGIN EDITABLE INCLUDES
// add extra device-side includes here, e.g. <cuda_pipeline.h>, <cooperative_groups.h>
// END EDITABLE INCLUDES

using namespace std;
using namespace nvcuda;

// BEGIN EDITABLE KERNEL
// edit this block freely. add __device__ helpers above ``kernel`` if needed.
// the LAUNCH block below must stay consistent with whatever signature you use.
__global__ void kernel(int dim_m, int dim_n, int dim_k, float *d_a, float *d_b,
                       float *d_c) {
  int offset_a_m = 64 * blockIdx.x;
  int offset_b_n = 64 * blockIdx.y;
  int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  __shared__ half block_a[16][64];
  __shared__ half block_b[16][64];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += 16) {
    __syncthreads();
    for (int j = 0; j < 16; ++j) {
      block_a[j][i] = __float2half(d_a[(k + j) * dim_m + offset_a_m + i]);
      block_b[j][i] = __float2half(d_b[(offset_b_n + i) * dim_k + k + j]);
    }
    __syncthreads();
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[0][row_tile * 16], 64);
      for (int c = 0; c < 4; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
            b_frag;
        wmma::load_matrix_sync(b_frag, &block_b[0][c * 16], 64);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }
  }
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 4; c++) {
      int c_m = offset_a_m + (warp_id * 2 + r) * 16;
      int c_n = offset_b_n + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m,
                                wmma::mem_col_major);
    }
  }
}
// END EDITABLE KERNEL

int main(int argc, const char **argv) {
  // When --profile-only is passed (the harness profiler does this under ncu),
  // skip the cuBLAS reference loop so the profiler only sees the candidate kernel.
  // Correctness verification + cublas timing are also skipped in this mode.
  bool profile_only = false;
  for (int i = 1; i < argc; i++) {
    if (argv[i] && std::string(argv[i]) == "--profile-only") profile_only = true;
  }
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C_ref, *C_cand;
  cudaMallocManaged(&A, (size_t)m * k * sizeof(float));
  cudaMallocManaged(&B, (size_t)k * n * sizeof(float));
  cudaMallocManaged(&C_ref, (size_t)m * n * sizeof(float));
  cudaMallocManaged(&C_cand, (size_t)m * n * sizeof(float));
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      A[(size_t)k * i + j] = drand48();
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      B[(size_t)n * i + j] = drand48();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      C_ref[(size_t)m * i + j] = C_cand[(size_t)m * i + j] = 0;

  int64_t num_flops =
      (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = 0.0;
  double cublas_gflops = 0.0;
  cublasHandle_t cublas_handle;
  if (!profile_only) {
    cublasCreate(&cublas_handle);
    // cuBLAS reference: 2 warmup iters, then ``Nt`` timed iters averaged
    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt + 2; i++) {
      if (i == 2)
        tic = chrono::steady_clock::now();
      cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A,
                   CUDA_R_32F, m, B, CUDA_R_32F, k, &beta, C_ref, CUDA_R_32F, m,
                   CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    cublas_gflops = double(num_flops) / tcublas / 1.0e9;
  }
  // Candidate kernel: 2 warmup iters, then ``Nt`` timed iters averaged
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2)
      tic = chrono::steady_clock::now();
    // BEGIN EDITABLE LAUNCH
    int tile = 64;
    dim3 block = dim3(tile);
    dim3 grid = dim3((m + tile - 1) / tile, (n + tile - 1) / tile);
    kernel<<<grid, block>>>(m, n, k, A, B, C_cand);
    // END EDITABLE LAUNCH
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  double tcandidate = chrono::duration<double>(toc - tic).count() / Nt;
  double candidate_gflops = double(num_flops) / tcandidate / 1.0e9;
  double mean_err = 0.0;
  if (!profile_only) {
    double err = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        err += fabs(C_ref[(size_t)m * i + j] - C_cand[(size_t)m * i + j]);
      }
    }
    mean_err = err / double((size_t)n * m);
    printf("CUBLAS: %.2f Gflops, CANDIDATE: %.2f Gflops\n", cublas_gflops,
           candidate_gflops);
    printf("error: %lf\n", mean_err);
    printf("RESULT cublas_gflops=%.6f candidate_gflops=%.6f"
           " cublas_ms=%.6f candidate_ms=%.6f mean_abs_error=%.9f"
           " num_flops=%lld m=%d n=%d k=%d\n",
           cublas_gflops, candidate_gflops, tcublas * 1000.0, tcandidate * 1000.0,
           mean_err, (long long)num_flops, m, n, k);
    cublasDestroy(cublas_handle);
  }
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_ref);
  cudaFree(C_cand);
  return 0;
}
