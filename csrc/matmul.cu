#include "commons.cuh"
#include "ops.h"

#define TILE_WIDTH 16
#define COARSE_FACTOR 2
#define BLOCK_M (TILE_WIDTH*COARSE_FACTOR)
#define A_S_STRIDE (BLOCK_M + COARSE_FACTOR)

__global__ void matmultile_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int K, int N){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int t  = ty*TILE_WIDTH + tx;

    // Columns are strided by TILE_WIDTH so neighbouring threads (tx) still touch neighbouring columns -> coalesced
    const int blockRow = blockIdx.y*BLOCK_M;
    const int rowStart = blockRow + ty*COARSE_FACTOR;
    const int colStart = blockIdx.x*TILE_WIDTH*COARSE_FACTOR + tx;

    // A tile stored transposed: A_sT[k][m] = A[blockRow + m][tile*TILE_WIDTH + k]
    __shared__ __align__(16) float A_sT[TILE_WIDTH][A_S_STRIDE];
    // B_s holds all COARSE_FACTOR tiles of B (side by side along N) this block needs for one phase
    __shared__ float B_s[COARSE_FACTOR][TILE_WIDTH][TILE_WIDTH];

    // One accumulator per output this thread owns
    float sum[COARSE_FACTOR][COARSE_FACTOR];
    #pragma unroll
    for (int r = 0; r < COARSE_FACTOR; r++){
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++){
            sum[r][c] = 0.0f;
        }
    }

    // Walk along K one tile at a time
    const int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int tile = 0; tile < numTiles; tile++){
        // 1) Load this phase's tiles into shared memory (zero-pad out-of-bounds)
        #pragma unroll
        for (int l = 0; l < COARSE_FACTOR; l++){
            const int idx = t + l*TILE_WIDTH*TILE_WIDTH;
            const int m = idx / TILE_WIDTH;
            const int k = idx % TILE_WIDTH;
            const int row = blockRow + m;
            const int aCol = tile*TILE_WIDTH + k;
            A_sT[k][m] = (row < M && aCol < K) ? A[row*K + aCol] : 0.0f;
        }
        const int bRow = tile*TILE_WIDTH + ty;
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++){
            const int col = colStart + c*TILE_WIDTH;
            B_s[c][ty][tx] = (bRow < K && col < N) ? B[bRow*N + col] : 0.0f;
        }
        __syncthreads();

        // 2) Accumulate partial dot products: pull COARSE_FACTOR values of A (contiguous -> LDS.64) and of B into registers,
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++){
            float a_reg[COARSE_FACTOR], b_reg[COARSE_FACTOR];
            #pragma unroll
            for (int r = 0; r < COARSE_FACTOR; r++) a_reg[r] = A_sT[i][ty*COARSE_FACTOR + r];
            #pragma unroll
            for (int c = 0; c < COARSE_FACTOR; c++) b_reg[c] = B_s[c][i][tx];
            #pragma unroll
            for (int r = 0; r < COARSE_FACTOR; r++){
                #pragma unroll
                for (int c = 0; c < COARSE_FACTOR; c++){
                    sum[r][c] += a_reg[r] * b_reg[c];
                }
            }
        }
        __syncthreads();
    }

    // Write results back to C
    #pragma unroll
    for (int r = 0; r < COARSE_FACTOR; r++){
        const int row = rowStart + r;
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++){
            const int col = colStart + c*TILE_WIDTH;
            if (row < M && col < N){
                C[row*N + col] = sum[r][c];
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B){
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    if (M==0 || N==0) return C;

    dim3 numThreadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    const int tilePerBlock = TILE_WIDTH*COARSE_FACTOR;
    dim3 numBlocks((N+tilePerBlock-1)/tilePerBlock, (M+tilePerBlock-1)/tilePerBlock);

    auto stream = c10::cuda::getCurrentCUDAStream();
    matmultile_kernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
