#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"

#define TILE_WIDTH 16

__global__ void matmul_conv_fused(
    const __half *mask, 
    const __half *input, 
    __half *output,
    int Batch, 
    int Map_out, 
    int Channel, 
    int Height, 
    int Width, 
    int K
)
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

    __shared__ __half tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half tileB[TILE_WIDTH][TILE_WIDTH];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_unroll = Batch * Height_out * Width_out;
    const int H_unroll = Channel * K * K;

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    __half val = __float2half(0.0f);

    for (int tileId = 0; tileId < (H_unroll - 1) / TILE_WIDTH + 1; tileId++) {
        // load mask
        if (row < Map_out && tileId * TILE_WIDTH + tx < H_unroll) {
            tileA[ty][tx] = mask[row * H_unroll + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = __float2half(0.0f);
        }

        // load
        if (col < W_unroll && tileId * TILE_WIDTH + ty < H_unroll) {
            // load input
            int b = col / (Height_out * Width_out);
            int w_unroll = col % (Height_out * Width_out);

            int h_out = w_unroll / Width_out;
            int w_out = w_unroll % Width_out;

            int index = tileId * TILE_WIDTH + ty;
            int c = index / (K * K);
            int p = (index % (K * K)) / K;
            int q = index % K;

            tileB[ty][tx] = in_4d(b, c, h_out + p, w_out + q);
        } else {
            tileB[ty][tx] = __float2half(0.0f);
        }
        __syncthreads();

        if (row < Map_out && col < W_unroll) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val = __hadd(val, __hmul(tileA[ty][i], tileB[i][tx]));
            }
        }
        __syncthreads();
    }

    if (row < Map_out && col < W_unroll) {
        int b = col / (Height_out * Width_out);
        int x = col % (Height_out * Width_out);

        output[b * Map_out * (Height_out * Width_out) + row * (Height_out * Width_out) + x] = val;
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
    const int input_size = Batch * Channel * Height * Width;
    const int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);;
    const int mask_size = Map_out * Channel * K * K;

    // alloc memory
    cudaMalloc((void **)device_input_ptr, input_size * sizeof(__half));
    cudaMalloc((void **)device_output_ptr, output_size * sizeof(__half));
    cudaMalloc((void **)device_mask_ptr, mask_size * sizeof(__half));

    __half *input = (__half *)malloc(input_size * sizeof(__half));
    __half *mask = (__half *)malloc(mask_size * sizeof(__half));

    for (size_t i = 0; i < input_size; ++i) {
        input[i] = __float2half(host_input[i]);
    }
    for (size_t i = 0; i < mask_size; ++i) {
        mask[i] = __float2half(host_mask[i]);
    }

    cudaMemcpy(*device_input_ptr, input, input_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, mask, mask_size * sizeof(__half), cudaMemcpyHostToDevice);
    
    free(input);
    free(mask);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the fused kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_unrolled = Batch * Height_out * Width_out;

    // Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 matmul_grid_dim((Width_unrolled - 1) / TILE_WIDTH + 1, (Map_out - 1) / TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    matmul_conv_fused<<<matmul_grid_dim, matmul_block_dim>>>(
        (const __half*) device_mask, (const __half*) device_input, (__half*) device_output, Batch,
        Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    __half *output = (__half *)malloc(output_size * sizeof(__half));

    cudaMemcpy(output, device_output, output_size *sizeof(__half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < output_size; ++i) {
        host_output[i] = __half2float(output[i]);
    }

    // Free memory
    free(output);
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