#include <wb.h>

#define wbCheck(stmt)                                        \
  do                                                         \
  {                                                          \
    cudaError_t err = stmt;                                  \
    if (err != cudaSuccess)                                  \
    {                                                        \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
      wbLog(ERROR, "Failed to run stmt ", #stmt);            \
      return -1;                                             \
    }                                                        \
  } while (0)

//@@ Define any useful program-wide constants here
#define FILTER_DIM 3
#define TILE_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float filter_c[FILTER_DIM][FILTER_DIM][FILTER_DIM];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size)
{
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  unsigned int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  unsigned int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  unsigned int depth = blockIdx.z * TILE_WIDTH + threadIdx.z;

  __shared__ float tile[TILE_WIDTH + FILTER_DIM - 1][TILE_WIDTH + FILTER_DIM - 1][TILE_WIDTH + FILTER_DIM - 1];

  int row_i = row - (FILTER_DIM / 2);
  int col_i = col - (FILTER_DIM / 2);
  int depth_i = depth - (FILTER_DIM / 2);

  if ((row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (depth_i >= 0) && (depth_i < z_size)) {
    tile[tz][ty][tx] += input[(depth_i * TILE_WIDTH * y_size) + (row_i * x_size) + col_i];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }

  __syncthreads();

  float output_value = 0.0f;
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH) {
    for (int z = 0; z < FILTER_DIM; z++) {
      for (int y = 0; y < FILTER_DIM; y++) {
        for (int x = 0; x < FILTER_DIM; x++) {
          output_value += tile[tz + z][ty + y][tx + x] * filter_c[z][y][x];
        }
      }
    }
    if (row < y_size && col < x_size && depth < z_size) {
      output[(depth * y_size * x_size) + (row * x_size) + col] = output_value;
    }
  }
}

int main(int argc, char *argv[])
{
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  wbCheck(cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float)));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(filter_c, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice));

  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH + (FILTER_DIM - 1), TILE_WIDTH + (FILTER_DIM - 1), TILE_WIDTH + (FILTER_DIM - 1));
  dim3 dimGrid(ceil(x_size / (1.0 * TILE_WIDTH)), ceil(y_size / (1.0 * TILE_WIDTH)), ceil(z_size / (1.0 * TILE_WIDTH)));

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost));

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // for (int i = 0; i < 10; i++) {
  //   printf("%f ", hostOutput[i]);
  // }

  //@@ Free device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
