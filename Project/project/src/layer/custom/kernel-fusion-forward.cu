#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.

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

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_unroll = Batch * Height_out * Width_out;
    const int H_unroll = Channel * K * K;

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty; 
    int col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (H_unroll - 1) / TILE_WIDTH + 1; tileId++) {
        // load mask
        if (row < Map_out && tileId * TILE_WIDTH + tx < H_unroll) {
            tileA[ty][tx] = mask[row * H_unroll + tileId * TILE_WIDTH + threadIdx.x];
        } else {
            tileA[ty][tx] = 0;
        }

        // load 
        if (col < W_unroll && tileId * TILE_WIDTH + ty < H_unroll) {
            // load input
            size_t b = col / (Height_out * Width_out);
            size_t w_unroll = col % (Height_out * Width_out);
            
            size_t h_out = w_unroll / Width_out;
            size_t w_out = w_unroll % Width_out;

            size_t index = tileId * TILE_WIDTH + ty;
            size_t c = index / (K * K);
            size_t p = (index % (K * K)) / K;
            size_t q = index % K;
        
            tileB[ty][tx] = in_4d(b, c, h_out + p, w_out + q);
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < Map_out && col < W_unroll) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                if (i < TILE_WIDTH)
                    val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < Map_out && col < W_unroll) {
        size_t b = col / (Height_out * Width_out);
        size_t x = col % (Height_out * Width_out); 

        output[b * Map_out * (Height_out * Width_out) + row * (Height_out * Width_out) + x] = val;
    }

    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

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
    // TODO: Set the kernel dimensions and call the fused kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_unrolled = Batch * Height_out * Width_out;

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 matmul_grid_dim((Width_unrolled - 1) / TILE_WIDTH + 1, (Map_out - 1) / TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    matmul_conv_fused<<<matmul_grid_dim, matmul_block_dim>>>(
        device_mask, device_input, device_output, Batch, 
        Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // TODO: Free device memory
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