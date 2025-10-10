#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define wbCheck(stmt)                                      \
    cudaError_t err = cudaGetLastError();                  \
    if (err != cudaSuccess)                                \
    {                                                      \
        std::cout << "CUDA error after: " << #stmt << "\n" \
                  << cudaGetErrorString(err) << std::endl; \
        exit(-1);                                          \
    }

#define TILE_WIDTH 16

// Adapted from Chapter 16 "Kernel for the forward path of a convolutional layer" from Programming Massively Parallel Processors, 4th Edition
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    const int W_GRID = (Width_out - 1) / TILE_WIDTH + 1;

    int m = blockIdx.x;     // feature_map
    int b = blockIdx.z;     // batch
    int h = (blockIdx.y / W_GRID) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_GRID) * TILE_WIDTH + threadIdx.x;

    // boundary check
    if (w < Width_out && h < Height_out) {
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(b, m, h, w) = acc;
    }

#undef out_4d
#undef in_4d
#undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // define sizes
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

// Adapted from Chapter 16 "Host code for launching a convolutional layer kernel" from Programming Massively Parallel Processors, 4th Edition
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_GRID = ((Height - K + 1) + 1) / TILE_WIDTH + 1;
    const int W_GRID = ((Width - K + 1) + 1) / TILE_WIDTH + 1;
    const int Y = H_GRID * W_GRID;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Map_out, Y, Batch);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
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