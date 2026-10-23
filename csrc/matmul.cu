#include "commons.cuh"
#include "ops.h"

// 16x16 = 256 threads/block: up to 6 blocks per SM on sm_86, and enough blocks for small MLP layers
#define TILE_WIDTH 16
#define COARSE_FACTOR 1

// C (M x N) = A (M x K) @ B (K x N), all row-major
__global__ void matmultile_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int K, int N){
    int row = blockIdx.y*blockDim.y + threadIdx.y;                      // over M
    int colStart = blockIdx.x*blockDim.x*COARSE_FACTOR + threadIdx.x;   // over N

    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    // B_s holds all COARSE_FACTOR tiles of B this block needs for one phase
    __shared__ float B_s[COARSE_FACTOR][TILE_WIDTH][TILE_WIDTH];

    // Register tile: the COARSE_FACTOR outputs this thread owns
    // Loops over COARSE_FACTOR must be unrolled so sum[] stays in registers
    float sum[COARSE_FACTOR];
    #pragma unroll
    for (int j = 0; j<COARSE_FACTOR; j++){
        sum[j] = 0.0f;
    }
    for (int tile = 0; tile<(K+TILE_WIDTH-1)/TILE_WIDTH; tile++){        // loop over K
        int aCol = tile*TILE_WIDTH + threadIdx.x;
        int bRow = tile*TILE_WIDTH + threadIdx.y;
        A_s[threadIdx.y][threadIdx.x] = (row<M && aCol<K) ? A[row*K + aCol] : 0.0f;
        #pragma unroll
        for (int c = 0; c<COARSE_FACTOR; c++){
            int col = colStart + c*TILE_WIDTH;
            B_s[c][threadIdx.y][threadIdx.x] = (bRow<K && col<N) ? B[bRow*N + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for(int i = 0; i<TILE_WIDTH;i++){
            // Load A once into a register, reuse it for every output column
            float a = A_s[threadIdx.y][i];
            #pragma unroll
            for (int c = 0; c<COARSE_FACTOR; c++){
                sum[c] += a*B_s[c][i][threadIdx.x];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int c = 0; c<COARSE_FACTOR;c++){
        int col = colStart + c*TILE_WIDTH;
        if (row<M && col<N){
            C[row*N+col] = sum[c];
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B){
    // A and B are already on the GPU, so no cudaMalloc/cudaMemcpy; PyTorch owns all the memory
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    if (M==0 || N==0) return C;

    dim3 numThreadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    // Each block covers TILE_WIDTH*COARSE_FACTOR columns, so fewer blocks in x
    dim3 numBlocks((N+numThreadsPerBlock.x*COARSE_FACTOR-1)/(numThreadsPerBlock.x*COARSE_FACTOR), (M+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);

    auto stream = c10::cuda::getCurrentCUDAStream();
    matmultile_kernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
