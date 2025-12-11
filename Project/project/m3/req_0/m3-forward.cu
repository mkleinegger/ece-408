#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define UNROLL_BLOCK_SIZE 256
#define N_STREAMS 4

__global__ void matrix_unrolling_kernel(
    const float *input, float *output,
    const int Batch, const int Channel,
    const int Height, const int Width,
    const int K)
{
    /*
    input  - input
    output - output
    Batch  - batch_size (number of images in x)
    Channel - number of input feature maps
    Height  - input height dimension
    Width   - input width dimension
    K       - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int W_unroll   = Height_out * Width_out;

#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + (i0)]
#define out_3d(i2, i1, i0) output[(i2) * (Batch * W_unroll) + (i1) * (W_unroll) + (i0)]

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (b < Batch && t < Channel * W_unroll)
    {
        int c        = t / W_unroll;
        int w_unroll = t % W_unroll;
        int h_out    = w_unroll / Width_out;
        int w_out    = w_unroll % Width_out;

        int w_base = c * K * K;
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                int h_unroll = w_base + p * K + q;
                out_3d(h_unroll, b, w_unroll) =
                    in_4d(b, c, h_out + p, w_out + q);
            }
        }
    }

#undef in_4d
#undef out_3d
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size)
{
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < Batch && x < image_size)
    {
        for (int m = 0; m < Map_out; m++)
        {
            output[b * Map_out * image_size + m * image_size + x] =
                input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_output,
    const float *host_input,
    const float *host_mask,
    float **device_output_ptr,
    float **device_input_ptr,
    float **device_mask_ptr,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K
)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    const size_t input_size  = (size_t)Batch * Channel * Height * Width * sizeof(float);
    const size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    const size_t mask_size   = (size_t)Map_out * Channel * K * K * sizeof(float);

    // alloc memory
    cudaMalloc((void **)device_input_ptr,  input_size);
    cudaMalloc((void **)device_output_ptr, output_size);
    cudaMalloc((void **)device_mask_ptr,   mask_size);

    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled  = Batch * Height_out * Width_out; // total width for all batches

    float *unrolled_matrix; // Height_unrolled x (Batch*Height_out*Width_out)
    float *matmul_output;   // Map_out x (Batch*Height_out*Width_out)
    cudaMalloc((void **)&unrolled_matrix, (size_t)Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void **)&matmul_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));

    cudaHostRegister((void*)host_input,  (size_t) input_size,  0);
    cudaHostRegister((void*)host_output, (size_t) output_size, 0);

    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; ++i)
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

    // Base per-stream batch; last stream may take the remainder
    const int batchPerStream = Batch / N_STREAMS;

    // Async copies per stream
    for (int i = 0; i < N_STREAMS; ++i)
    {
        size_t batchToStream = min(Batch - i * batchPerStream, batchPerStream);
        if (batchToStream <= 0) continue;
        size_t device_input_offset = (size_t)i * batchPerStream * Channel * Height * Width;

        cudaMemcpyAsync(
            *device_input_ptr + device_input_offset,
            host_input        + device_input_offset,
            (size_t)batchToStream * Channel * Height * Width * sizeof(float),
            cudaMemcpyHostToDevice,
            stream[i]
        );
    }

    // Unroll, matmul, permute per stream
    for (int i = 0; i < N_STREAMS; ++i)
    {
        size_t batchToStream = min(Batch - i * batchPerStream, batchPerStream);
        if (batchToStream <= 0) continue;
        size_t device_input_offset = (size_t)i * batchPerStream * Channel * Height * Width;
        size_t unrolled_matrix_offset = (size_t)i * batchPerStream * Channel * K * K * Height_out * Width_out;
        size_t device_output_offset = (size_t)i * batchPerStream * Map_out * Height_out * Width_out;
        size_t matmul_output_offset = (size_t)i * batchPerStream * Map_out * Height_out * Width_out;

        // Unrolling kernel
        int num_threads = Channel * Height_out * Width_out;
        int num_blocks  = (num_threads + UNROLL_BLOCK_SIZE - 1) / UNROLL_BLOCK_SIZE;

        dim3 unrolling_block_dim(UNROLL_BLOCK_SIZE, 1, 1);
        dim3 unrolling_grid_dim(num_blocks, batchToStream, 1);

        matrix_unrolling_kernel<<<unrolling_grid_dim, unrolling_block_dim, 0, stream[i]>>>(
            *device_input_ptr + device_input_offset,
            unrolled_matrix   + unrolled_matrix_offset,
            batchToStream,
            Channel,
            Height,
            Width,
            K
        );

        // Matrix multiplication
        dim3 matmul_grid_dim((((size_t) batchToStream * Height_out * Width_out) - 1) / MATMUL_TILE_WIDTH + 1,
                         (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
        dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
        matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim, 0, stream[i]>>>(
            *device_mask_ptr,
            unrolled_matrix + unrolled_matrix_offset,
            matmul_output   + matmul_output_offset,
            Map_out,
            Height_unrolled,
            Height_unrolled,
            batchToStream * Height_out * Width_out,
            Map_out,
            batchToStream * Height_out * Width_out
        );

        // Permute the result of matrix multiplication
        const int out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim(
            (out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1,
            batchToStream,
            1
        );

        matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE, 0, stream[i]>>>(
            matmul_output   + matmul_output_offset,
            *device_output_ptr + device_output_offset,
            Map_out,
            batchToStream,
            out_image_size
        );
    }

    for (int i = 0; i < N_STREAMS; ++i)
    {
        size_t batchToStream = min(Batch - i * batchPerStream, batchPerStream);
        if (batchToStream <= 0) continue;
        size_t device_output_offset = (size_t)i * batchPerStream * Map_out * Height_out * Width_out;

        cudaMemcpyAsync(
            (float *)host_output + device_output_offset,
            *device_output_ptr + device_output_offset,
            (size_t)batchToStream * Map_out * Height_out * Width_out * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream[i]
        );
    }

    for (int i = 0; i < N_STREAMS; ++i)
        cudaStreamSynchronize(stream[i]);

    for (int i = 0; i < N_STREAMS; ++i)
        cudaStreamDestroy(stream[i]);

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
    cudaFree(*device_input_ptr);
    cudaFree(*device_output_ptr);
    cudaFree(*device_mask_ptr);
}

__host__ void GPUInterface::conv_forward_gpu(
    float *device_output,
    const float *device_input,
    const float *device_mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K
)
{
    // Not used in this setup (everything is done in prolog).
}

__host__ void GPUInterface::conv_forward_gpu_epilog(
    float *host_output,
    float *device_output,
    float *device_input,
    float *device_mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K
)
{
    // Not used in this setup (everything is done in prolog).
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
        std::cout << "Max block dimensions: "
                  << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, "
                  << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: "
                  << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, "
                  << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}