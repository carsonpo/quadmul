#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/pipeline>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "helpers.cu"
#include <iostream>
#include <vector>
#include <cstdint>
#include "configs.cu"
#include "cublas_v2.h"

#define div_ru(a, b) (((a) + (b) - 1) / (b))

#define WARP_SIZE 32
#define DEBUG false

// M is not constexpr-d because tokens * batch can vary, but the rest of the problem size is fixed for specific configs
template <int BlockRowWarps, int BlockColWarps, int WarpRowTiles, int WarpColTiles, int PatchM, int PatchN, int ChunkK, int NumStages, int PipelineStrategy, int kWMMA_M, int kWMMA_N, int kWMMA_K, int kN, int kK>
struct IGemmConfig
{
    static constexpr int kBlockRowWarps = BlockRowWarps;
    static constexpr int kBlockColWarps = BlockColWarps;
    static constexpr int kWarpRowTiles = WarpRowTiles;
    static constexpr int kWarpColTiles = WarpColTiles;
    static constexpr int kChunkK = ChunkK;
    static constexpr int kNumStages = NumStages;
    static constexpr int kPipelineStrategy = PipelineStrategy;

    static constexpr int kPatchM = PatchM;
    static constexpr int kPatchN = PatchN;

    // Derived constants
    static constexpr int kBlockRowTiles = kWarpRowTiles * kBlockRowWarps;
    static constexpr int kBlockColTiles = kWarpColTiles * kBlockColWarps;

    static constexpr int kTileSizeM = kWMMA_M * kBlockRowTiles;
    static constexpr int kTileSizeN = kWMMA_N * kBlockColTiles;
    static constexpr int kTileSizeK = kWMMA_K * kChunkK;

    static constexpr int K = kK;
    static constexpr int N = kN;
    static constexpr int WMMA_M = kWMMA_M;
    static constexpr int WMMA_N = kWMMA_N;
    static constexpr int WMMA_K = kWMMA_K;
};

// 128-bit vector type for efficient memory loads
using Data128B = int4;
using Data64B = int2;
constexpr int ALIGN_SIZE_A = 32;
constexpr int ALIGN_SIZE_B = 32;
#define PRESHUFFLE false

using namespace nvcuda;
using I4 = wmma::experimental::precision::s4;

template <typename Config>
__global__ void igemm(const uint8_t *A, const uint8_t *B, int32_t *C, int M)
{
    extern __shared__ int8_t shared_memory[];

    using FragA = wmma::fragment<wmma::matrix_a, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, I4, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, I4, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, int32_t>;

    // Set up shared memory tensors for A and B with multiple stages
    SmemTensor3D<uint8_t, Config::kNumStages, Config::kTileSizeM, Config::kTileSizeK / 2>
        smemA(shared_memory);
    SmemTensor3D<uint8_t, Config::kNumStages, Config::kTileSizeN, Config::kTileSizeK / 2> smemB(smemA.endPtr);

    // Set up global memory tensors for A, B, and C
    GMemTensor2D<uint8_t> gmemA((uint8_t *)A, M, Config::K / 2);
    GMemTensor2D<uint8_t> gmemB((uint8_t *)B, Config::N, Config::K / 2); // Note: B is transposed
    GMemTensor2D<int32_t> gmemC(C, M, Config::N);

    // Calculate warp and lane IDs
    int warp_id = threadIdx.x / WARP_SIZE;
    // int warp_row = warp_id / Config::kBlockColWarps;
    // int warp_col = warp_id % Config::kBlockColWarps;

    int warp_row = warp_id / (Config::kBlockColWarps / Config::kPatchN);
    int warp_col = warp_id % (Config::kBlockColWarps / Config::kPatchN);

    // Calculate starting positions for this block
    int block_row_start = blockIdx.x * Config::kTileSizeM;
    int block_col_start = blockIdx.y * Config::kTileSizeN;

    FragA a_frag[Config::kPatchM][Config::kWarpRowTiles];
    FragB b_frag[Config::kPatchN][Config::kWarpColTiles];
    FragC c_frag[Config::kPatchM][Config::kPatchN][Config::kWarpRowTiles][Config::kWarpColTiles];

    // Initialize accumulator fragments

    for (int pm = 0; pm < Config::kPatchM; pm++)
    {

        for (int pn = 0; pn < Config::kPatchN; pn++)
        {

            for (int i = 0; i < Config::kWarpRowTiles; i++)
            {

                for (int j = 0; j < Config::kWarpColTiles; j++)
                {
                    wmma::fill_fragment(c_frag[pm][pn][i][j], 0);
                }
            }
        }
    }

    constexpr int EffectiveTileSizeK = Config::kTileSizeK / 2;

    auto load_A_tile = [&](int stage, int k_offset)
    {
        for (int i = threadIdx.x; i < (Config::kTileSizeM * Config::kTileSizeK) / ALIGN_SIZE_A; i += blockDim.x)
        {
            int row = (i * ALIGN_SIZE_A) / Config::kTileSizeK;
            int col = (i * ALIGN_SIZE_A) % Config::kTileSizeK;
            int global_row = block_row_start + row;
            int global_col = k_offset + col;
            if (global_row < M && global_col + ALIGN_SIZE_A <= Config::K)
            {
                uint8_t *shared_ptr = smemA.get_ptr(stage, row, col / 2);
                uint8_t *global_ptr = gmemA.get_ptr(global_row, global_col / 2);
                __pipeline_memcpy_async(shared_ptr, global_ptr, ALIGN_SIZE_A / 2);
            }
        }
    };

    // Lambda for loading B tiles
    auto load_B_tile = [&](int stage, int k_offset)
    {
        for (int i = threadIdx.x; i < (Config::kTileSizeN * Config::kTileSizeK) / ALIGN_SIZE_B; i += blockDim.x)
        {
            int row = (i * ALIGN_SIZE_B) / Config::kTileSizeK;
            int col = (i * ALIGN_SIZE_B) % Config::kTileSizeK;
            int global_row = block_col_start + row;
            int global_col = k_offset + col;

            if (global_row < Config::N && global_col + ALIGN_SIZE_B <= Config::K)
            {
                uint8_t *shared_ptr = smemB.get_ptr(stage, row, col / 2);
                uint8_t *global_ptr = gmemB.get_ptr(global_row, global_col / 2);
                __pipeline_memcpy_async(shared_ptr, global_ptr, ALIGN_SIZE_B / 2);
            }
        }
    };

    // Lambda for storing C tiles
    auto store_C_tile = [&]()
    {
        for (int pm = 0; pm < Config::kPatchM; pm++)
        {

            for (int pn = 0; pn < Config::kPatchN; pn++)
            {

                for (int i = 0; i < Config::kWarpRowTiles; i++)
                {

                    for (int j = 0; j < Config::kWarpColTiles; j++)
                    {
                        int row = block_row_start + ((warp_row * Config::kPatchM + pm) * Config::kWarpRowTiles + i) * Config::WMMA_M;
                        int col = block_col_start + ((warp_col * Config::kPatchN + pn) * Config::kWarpColTiles + j) * Config::WMMA_N;

                        if (row < M && col < Config::N)
                        {
                            wmma::store_matrix_sync(
                                gmemC.get_ptr(row, col),
                                c_frag[pm][pn][i][j],
                                Config::N,
                                wmma::mem_row_major);
                        }
                    }
                }
            }
        }
        __syncthreads();
    };

    auto pipeline_strategy_1 = [&]()
    {
        // Load first stage
        load_A_tile(0, 0);
        load_B_tile(0, 0);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        int current_stage = 0;
        for (int k = 0; k < Config::K; k += Config::kTileSizeK)
        {
            // Start loading next stage if available
            if (k + Config::kTileSizeK < Config::K)
            {
                int next_stage = 1 - current_stage;
                load_A_tile(next_stage, k + Config::kTileSizeK);
                load_B_tile(next_stage, k + Config::kTileSizeK);
                __pipeline_commit();
            }

            // Compute using current stage
            for (int kk = 0; kk < Config::kTileSizeK; kk += Config::WMMA_K)
            {

                for (int pm = 0; pm < Config::kPatchM; pm++)
                {

                    for (int i = 0; i < Config::kWarpRowTiles; i++)
                    {
                        wmma::load_matrix_sync(
                            a_frag[pm][i],
                            smemA.get_ptr(current_stage, (warp_row * Config::kPatchM + pm) * Config::kWarpRowTiles * Config::WMMA_M + i * Config::WMMA_M, kk / 2),
                            Config::kTileSizeK);
                    }
                }

                for (int pn = 0; pn < Config::kPatchN; pn++)
                {

                    for (int j = 0; j < Config::kWarpColTiles; j++)
                    {
                        wmma::load_matrix_sync(
                            b_frag[pn][j],
                            smemB.get_ptr(current_stage, (warp_col * Config::kPatchN + pn) * Config::kWarpColTiles * Config::WMMA_N + j * Config::WMMA_N, kk / 2),
                            Config::kTileSizeK);
                    }
                }

                for (int pm = 0; pm < Config::kPatchM; pm++)
                {

                    for (int pn = 0; pn < Config::kPatchN; pn++)
                    {

                        for (int i = 0; i < Config::kWarpRowTiles; i++)
                        {

                            for (int j = 0; j < Config::kWarpColTiles; j++)
                            {
                                wmma::mma_sync(c_frag[pm][pn][i][j], a_frag[pm][i], b_frag[pn][j], c_frag[pm][pn][i][j]);
                            }
                        }
                    }
                }
            }

            // Wait for next stage to finish loading
            __pipeline_wait_prior(0);
            __syncthreads();

            // Swap stages
            current_stage = 1 - current_stage;
        }
    };

    pipeline_strategy_1();

    // Store results
    store_C_tile();
}

template <typename Config>
void launch_igemm(const uint8_t *A, const uint8_t *B, int32_t *C, int M, cudaStream_t stream)
{
    dim3 grid_dim(div_ru(M, Config::kTileSizeM), div_ru(Config::N, Config::kTileSizeN));
    dim3 block_dim(WARP_SIZE * (Config::kBlockRowWarps / Config::kPatchM) * (Config::kBlockColWarps / Config::kPatchN));

    // printf("grid_dim x: %d, block_dim x: %d, grid_dim y: %d, block_dim y: %d\n", grid_dim.x, block_dim.x, grid_dim.y, block_dim.y);
    // printf("M: %d, N: %d, K: %d\n", M, Config::N, Config::K);

    size_t shared_mem_size = Config::kNumStages * (Config::kTileSizeM * Config::kTileSizeK * sizeof(int8_t) / 2 + Config::kTileSizeN * Config::kTileSizeK * sizeof(int8_t) / 2);

    // printf("shared_mem_size: %lu\n", shared_mem_size);

    igemm<Config><<<grid_dim, block_dim, shared_mem_size, stream>>>(A, B, C, M);
    // cudaDeviceSynchronize();

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }
}

void cpu_gemm_int8(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    // return;
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[n * K + k]);
            }
            C[m * N + n] = sum;
        }
    }
}

bool compare_results(const int32_t *gpu_result, const int32_t *cpu_result, int size, float tolerance = 1e-5)
{
    for (int i = 0; i < size; ++i)
    {
        if (std::abs(static_cast<float>(gpu_result[i] - cpu_result[i])) > tolerance)
        {
            std::cout << "Mismatch at index " << i << ": GPU = " << gpu_result[i] << ", CPU = " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

#define LAUNCH_KERNEL_IF_CONDITION(config, mCond, nCond, kCond)                        \
    else if (n == nCond && m == mCond && k == kCond)                                   \
    {                                                                                  \
        using ThisConfig = IGemmConfig<config.BlockRowWarps, config.BlockColWarps,     \
                                       config.WarpRowTiles, config.WarpColTiles,       \
                                       config.PatchM, config.PatchN, config.ChunkK,    \
                                       config.NumStages, config.PipelineStrategy,      \
                                       config.kWMMA_M, config.kWMMA_N, config.kWMMA_K, \
                                       config.N, config.K>;                            \
        launch_igemm<ThisConfig>(A_ptr, B_ptr, C_ptr, m, stream);                      \
        return;                                                                        \
    }

void wrapper(void *A, void *B, void *C, const int m, const int n, const int k, cudaStream_t stream)
{
    const uint8_t *A_ptr = reinterpret_cast<const uint8_t *>(A);
    const uint8_t *B_ptr = reinterpret_cast<const uint8_t *>(B);
    int32_t *C_ptr = reinterpret_cast<int32_t *>(C);

    if (false)
    {
    }
    LAUNCH_KERNEL_IF_CONDITION(quadmul_4096_57344_8192, 4096, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_4096_8192_8192, 4096, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_4096_28672_4096, 4096, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_4096_10240_8192, 4096, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_4096_6144_4096, 4096, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_2048_8192_28672, 2048, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_2048_10240_8192, 2048, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_2048_8192_8192, 2048, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_2048_28672_4096, 2048, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_256_10240_8192, 256, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_256_4096_14336, 256, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_1024_6144_4096, 1024, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_256_6144_4096, 256, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_256_28672_4096, 256, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_128_10240_8192, 128, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_256_57344_8192, 256, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_1024_28672_4096, 1024, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_1024_4096_14336, 1024, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_128_8192_8192, 128, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_4096_4096_14336, 4096, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_128_28672_4096, 128, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_512_6144_4096, 512, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_128_57344_8192, 128, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_512_8192_28672, 512, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_1024_10240_8192, 1024, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_128_8192_28672, 128, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_128_6144_4096, 128, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_2048_6144_4096, 2048, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_2048_4096_14336, 2048, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_512_57344_8192, 512, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_128_4096_14336, 128, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_256_8192_28672, 256, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_1024_8192_28672, 1024, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_512_28672_4096, 512, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_4096_8192_28672, 4096, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_512_4096_14336, 512, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_512_8192_8192, 512, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_512_10240_8192, 512, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_1024_8192_8192, 1024, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_2048_57344_8192, 2048, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_256_8192_8192, 256, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(quadmul_1024_57344_8192, 1024, 57344, 8192)
}

#define U4_TO_S4(x) (x > 7 ? (int8_t)(x) - 16 : x)

void unpack_int4_matrix(uint8_t *B, int8_t *B_unpacked, int N, int K)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; j += 2)
        {
            B_unpacked[i * K + j] = ((B[i * K / 2 + j / 2]) & 0b00001111);
            B_unpacked[i * K + j + 1] = (((B[i * K / 2 + j / 2] >> 4) & 0b00001111));

            B_unpacked[i * K + j] = U4_TO_S4(B_unpacked[i * K + j]);
            B_unpacked[i * K + j + 1] = U4_TO_S4(B_unpacked[i * K + j + 1]);

            // B_unpacked[(i + 0) * K + j] = (int8_t)((B[i * K / 2 + j / 2]) & 0b00001111);
            // B_unpacked[(i + 1) * K + j] = (int8_t)(((B[(i + 1) * K / 2 + j / 2] >> 4) & 0b00001111));

            // B_unpacked[(i + 0) * K + j] = U4_TO_S4(B_unpacked[(i + 0) * K + j / 2]);
            // B_unpacked[(i + 1) * K + j] = U4_TO_S4(B_unpacked[(i + 1) * K + j / 2]);
        }
    }
}

// int main()
// {
//     // Choose matrix dimensions (multiples of 16 for best performance)
//     // test_packed_ldmatrix_format<<<1, 32>>>();
//     // test_ldmatrix_format<<<1, 32>>>();
//     // cudaDeviceSynchronize();
//     // return 0;

//     const int M = 1024, N = 8192, K = 8192;

//     /* int BlockRowWarps, int BlockColWarps, int WarpRowTiles, int WarpColTiles, int ChunkK, int NumStages, int PipelineStrategy, int kWMMA_M, int kWMMA_N, int kWMMA_K, int kN, int kK*/

//     // using Config = IGemmConfig<2, 6, 6, 2, 1, 2, 4, 2, 1, 16, 16, 16, N, K>;
//     // using Config = IGemmConfig<4, 4, 4, 6, 4, 1, 4, 2, 1, 8, 8, 32, N, K>;
//     using Config = IGemmConfig<4, 4, 2, 4, 4, 1, 2, 2, 1, 8, 8, 32, N, K>;
//     // using Config = IGemmConfig<2, 2, 4, 4, 2, 2, 3, 32, 8, 16, N, K>;
//     // using Config = IGemmConfig<3, 2, 3, 4, 1, 2, 2, 16, 16, 16, N, K>;
//     // using Config = IGemmConfig<1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 32, N, K>;

//     // Allocate host memory
//     std::vector<uint8_t> h_A(M * K / 2);
//     std::vector<uint8_t> h_B(N * K / 2);
//     std::vector<int8_t> h_B_unpacked(N * K);
//     std::vector<int8_t> h_A_unpacked(M * K);
//     std::vector<int32_t> h_C_gpu(M * N);
//     std::vector<int32_t> h_C_cpu(M * N);

//     // Initialize input matrices with random data
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, 255);

//     std::generate(h_A.begin(), h_A.end(), [&]()
//                   { return static_cast<uint8_t>(dis(gen)); });
//     std::generate(h_B.begin(), h_B.end(), [&]()
//                   { return static_cast<uint8_t>(dis(gen)); });

//     unpack_int4_matrix(h_B.data(), h_B_unpacked.data(), N, K);
//     unpack_int4_matrix(h_A.data(), h_A_unpacked.data(), M, K);

//     // reorganize_packed_B(h_B.data(), h_B_shfl.data(), N, K);

//     // Allocate device memory
//     uint8_t *d_A;
//     uint8_t *d_B;
//     // int8_t *d_B_shfl;
//     int32_t *d_C;
//     cudaMalloc(&d_A, M * K * sizeof(uint8_t) / 2);
//     cudaMalloc(&d_B, K * N * sizeof(uint8_t) / 2);
//     // cudaMalloc(&d_B_shfl, N * K * sizeof(int8_t) / 2);
//     cudaMalloc(&d_C, M * N * sizeof(int32_t));

//     // Copy input data to device
//     cudaMemcpy(d_A, h_A.data(), M * K * sizeof(uint8_t) / 2, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B.data(), K * N * sizeof(uint8_t) / 2, cudaMemcpyHostToDevice);

//     cudaDeviceSynchronize();

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     constexpr int numWarmups = 10;
//     constexpr int numTrials = 100;

//     for (int i = 0; i < numWarmups; ++i)
//     {
//         launch_igemm<Config>(d_A, d_B, d_C, M);
//     }

//     cudaDeviceSynchronize();

//     // Launch GPU kernel
//     cudaEventRecord(start);
//     for (int i = 0; i < numTrials; ++i)
//     {
//         launch_igemm<Config>(d_A, d_B, d_C, M);
//     }
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     // Calculate TOPS
//     double seconds = milliseconds / 1000.0;
//     double operations = static_cast<double>(M) * N * K * 2 * numTrials; // 2 ops per multiply-add
//     double tops = operations / (seconds * 1e12);

//     std::cout << "GPU Performance: " << tops << " TOPS" << std::endl;

//     // Copy result back to host
//     cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

//     // Compute CPU result (commented out for performance)
//     cpu_gemm_int8(h_A_unpacked.data(), h_B_unpacked.data(), h_C_cpu.data(), M, N, K);

//     // Compare results (commented out for performance)
//     bool results_match = compare_results(h_C_gpu.data(), h_C_cpu.data(), M * N);

//     if (results_match)
//     {
//         std::cout << "Results match! The WMMA GEMM implementation is correct." << std::endl;
//     }
//     else
//     {
//         std::cout << "Results do not match. There might be an error in the WMMA GEMM implementation." << std::endl;
//     }

//     // Clean up
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     return 0;
// }