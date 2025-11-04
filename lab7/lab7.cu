// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add_accumulated_block_sums(float *values, float *accBlockSum ,int len) {
  unsigned int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  if(blockIdx.x > 0) {
      if (i < len) values[i] += accBlockSum[blockIdx.x - 1];
      if ((i + blockDim.x) < len) values[i + blockDim.x] += accBlockSum[blockIdx.x - 1];
  }
}

__global__ void scan(float *input, float *output, int len, float *auxBlockSum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float XY[2 * BLOCK_SIZE];
  unsigned int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  XY[threadIdx.x] = 0.0f;
  XY[threadIdx.x + blockDim.x] = 0.0f;
  if (i < len) XY[threadIdx.x] = input[i];
  if (i + blockDim.x < len) XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];

  for (int stride = 1; stride < 2 * BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
      XY[index] += XY[index - stride];
    }
  }

  for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if ((index + stride) < 2 * BLOCK_SIZE) {
      XY[index + stride] += XY[index];
    }
  }

  __syncthreads();
  if (i < len) output[i] = XY[threadIdx.x];
  if (i + blockDim.x < len) output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];

  // assign blocksums
  if (auxBlockSum != NULL) {
    if(i + blockDim.x == len - 1) {
      auxBlockSum[blockIdx.x] = XY[threadIdx.x + blockDim.x];
    } else if(i == len - 1){
      auxBlockSum[blockIdx.x] = XY[threadIdx.x];
    } else if(threadIdx.x == BLOCK_SIZE - 1) {
      auxBlockSum[blockIdx.x] = XY[threadIdx.x + blockDim.x];  
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));

  int numOutputElements = numElements / (BLOCK_SIZE << 1);
  if (numElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }

  float *deviceAuxBlockSum;
  float *deviceAccBlockSum;

  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxBlockSum, numOutputElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAccBlockSum, numOutputElements * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceAuxBlockSum, 0, numOutputElements * sizeof(float)));
  wbCheck(cudaMemset(deviceAccBlockSum, 0, numOutputElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(numOutputElements, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, deviceAuxBlockSum);

  cudaDeviceSynchronize();

  scan<<<dimGrid, dimBlock>>>(deviceAuxBlockSum, deviceAccBlockSum, numOutputElements, NULL);

  cudaDeviceSynchronize();

  add_accumulated_block_sums<<<dimGrid, dimBlock>>>(deviceOutput, deviceAccBlockSum, numElements);
  
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));

  //@@  Free GPU Memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));
  wbCheck(cudaFree(deviceAuxBlockSum));
  wbCheck(cudaFree(deviceAccBlockSum));

  wbSolution(args, hostOutput, numElements);
  
  free(hostInput);
  free(hostOutput);

  return 0;
}

