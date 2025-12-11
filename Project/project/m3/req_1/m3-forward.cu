#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>
using namespace nvcuda;

#define TILE_WIDTH 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define WARP_SIZE 32

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    __shared__ float tileA[WMMA_M][WMMA_K];
    __shared__ float tileB[WMMA_K][WMMA_N];
    __shared__ float tileC[WMMA_M][WMMA_N];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_unroll = Height_out * Width_out;
    const int H_unroll = Channel * K * K;
    const int b = blockIdx.z;

    int tileRow = blockIdx.y * WMMA_M;
    int tileCol = blockIdx.x * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,  wmma::precision::tf32, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,  wmma::precision::tf32, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    for (int k0 = 0; k0 < H_unroll; k0 += WMMA_K) {
        for (int i = 0; i < 4; i++) {
            int elem = threadIdx.x + i * 32;
            int rowWithinTile = elem / WMMA_K;
            int colWithinTile = elem % WMMA_K;

            tileA[rowWithinTile][colWithinTile] = 0;
            if ((tileRow + rowWithinTile) < Map_out && (k0 + colWithinTile) < H_unroll)
                tileA[rowWithinTile][colWithinTile] = mask[(tileRow + rowWithinTile) * H_unroll + (k0 + colWithinTile)];
        }

        for (int i = 0; i < 4; i++) {
            int elem = threadIdx.x + i * 32;
            int rowWithinTile = elem / WMMA_N;    
            int colWithinTile = elem % WMMA_N;    

            int r   = k0 + rowWithinTile;
            int col = tileCol + colWithinTile;

            tileB[rowWithinTile][colWithinTile] = 0.0f;
            if (r < H_unroll && col < W_unroll) {
                int h_out = col / Width_out;
                int w_out = col % Width_out;

                int index = r;
                int c = index / (K * K);
                int p = (index % (K * K)) / K;
                int q = (index % (K * K)) % K;

                tileB[rowWithinTile][colWithinTile] = in_4d(b, c, h_out + p, w_out + q);
            }
        }

        __syncthreads();
        wmma::load_matrix_sync(fragA, &tileA[0][0], WMMA_K);
        wmma::load_matrix_sync(fragB, &tileB[0][0], WMMA_N);
        wmma::mma_sync(fragC, fragA, fragB, fragC);
        __syncthreads();
    }

    wmma::store_matrix_sync(&tileC[0][0], fragC, TILE_WIDTH, wmma::mem_row_major);
    __syncthreads();

    for (int i = 0; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
        int elem = threadIdx.x + i;
        int rowWithinTile = elem / WMMA_N;           
        int colWithinTile = elem % WMMA_N;           

        int m = tileRow + rowWithinTile;             
        int col = tileCol + colWithinTile;

        if (m < Map_out && col < W_unroll) {
            int heightOut = col / Width_out;
            int widthOut = col % Width_out;

            output[b * Map_out * (Height_out * Width_out) + m * (Height_out * Width_out) + heightOut * Width_out + widthOut] = tileC[rowWithinTile][colWithinTile];
        }
    }


    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int input_size = Batch * Channel * Height * Width * sizeof(float);
    const int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    const int mask_size = Map_out * Channel * K * K * sizeof(float);

    // alloc memory
    cudaMalloc((void **)device_input_ptr, input_size);
    cudaMalloc((void **)device_output_ptr, output_size);
    cudaMalloc((void **)device_mask_ptr, mask_size);

    // copy data
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the fused kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_unrolled = Height_out * Width_out;

    dim3 matmul_block_dim(WARP_SIZE, 1, 1);
    dim3 matmul_grid_dim((Width_unrolled - 1) / WMMA_M + 1, (Map_out - 1) / WMMA_N + 1, Batch);
    matmul_conv_fused<<<matmul_grid_dim, matmul_block_dim>>>(
        device_mask, device_input, device_output, Batch,
        Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}