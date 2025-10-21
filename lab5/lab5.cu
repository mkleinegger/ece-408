// Histogram Equalization

#include <wb.h>

#define wbCheck(stmt) do {                                            \
  cudaError_t err = stmt;                                             \
  if (err != cudaSuccess) {                                           \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
    return -1;                                                        \
  }                                                                   \
} while(0)

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 16
#define BLOCK_DIM HISTOGRAM_LENGTH

//@@ insert code here
__global__ void float_to_uchar_cast_kernel(float *input_image, unsigned char *output_image, int image_width, int image_height, int image_channels) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  if (col < image_width && row < image_height) {
    int idx = (row * image_width + col) * image_channels + channel;
    output_image[idx] =  (unsigned char)(255 * input_image[idx]);
  }
}

__global__ void uchar_to_float_cast_kernel(unsigned char *input_image, float *output_image, int image_width, int image_height, int image_channels) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  if (col < image_width && row < image_height) {
    int idx = (row * image_width + col) * image_channels + channel;
    output_image[idx] =  (float)(input_image[idx] / 255.0);
  }
}

__global__ void color_to_grayscale_kernel(unsigned char *color_image, unsigned char *grayscale_image, int image_width, int image_height, int image_channels) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col < image_width && row < image_height) {
    int grayscale_offset = row * image_width + col;
    int rgb_offset = grayscale_offset * image_channels;

    unsigned char r = color_image[rgb_offset];
    unsigned char g = color_image[rgb_offset + 1];
    unsigned char b = color_image[rgb_offset + 2];

    grayscale_image[grayscale_offset] = (unsigned char) (0.21f * r + 0.71f * g + 0.07f * b);
  }
}

__global__ void generate_hist_kernel(unsigned char *grayscale_image, unsigned int *hist, int image_width, int image_height) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col < image_width && row < image_height) {
      int value = grayscale_image[row * image_width + col];

      if (value >= 0 && value < 256) {
        atomicAdd(&(hist[value]), 1);
      }
  }
}

__device__ unsigned char apply_cdf_correction(unsigned char value, float *cdf, float cdf_min) {
  unsigned char corrected_value = 255 * (cdf[value] - cdf_min) / (1.0 - cdf_min);
  return min(max(corrected_value, 0), 255);
}

__global__ void equalize_image_kernel(unsigned char *image, unsigned char *corrected_image, float *cdf, int image_width, int image_height, int imageChannels) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  if (col < image_width && row < image_height) {
    int idx = (row * image_width + col) * imageChannels + channel;
    corrected_image[idx] = (unsigned char)(apply_cdf_correction(image[idx], cdf, cdf[0]));
  }
}

__global__ void generate_cdf_scan_kernel(unsigned int *hist, float *cdf, int image_width, int image_height) {
  __shared__ float buffer_s[BLOCK_DIM];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < HISTOGRAM_LENGTH) {
    buffer_s[threadIdx.x] = ((float) hist[idx] / (image_width * image_height));
  } else {
    buffer_s[threadIdx.x] = 0.0f;
  }

  for(unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2) {
    __syncthreads();
  
    float temp;
    if(threadIdx.x >= stride) {
      temp = buffer_s[threadIdx.x] + buffer_s[threadIdx.x - stride];
    }
    __syncthreads();
  
    if(threadIdx.x >= stride) {
        buffer_s[threadIdx.x] = temp;
    }
  }

  if (idx < HISTOGRAM_LENGTH) {
    cdf[idx] = buffer_s[threadIdx.x];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
  unsigned char *deviceUCharImage; 
  unsigned char *deviceGrayScaleImage;
  unsigned char *deviceCorrectedImage;
  unsigned int *deviceHistogram;
  float *deviceCDF;
  float *deviceOutput;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ insert code here
  int numTotalPixels = imageWidth * imageHeight * imageChannels;
  int numPixels = imageWidth * imageHeight;

  wbCheck(cudaMalloc((void **)&deviceInputImage, numTotalPixels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceUCharImage, numTotalPixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrayScaleImage, numPixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCorrectedImage, numTotalPixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numTotalPixels * sizeof(float)));

  wbCheck(cudaMemcpy(deviceInputImage, hostInputImageData, numTotalPixels * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int))); // for initcheck tests

  dim3 dimBlockConversion(TILE_WIDTH, TILE_WIDTH, imageChannels);
  dim3 dimGridConversion(ceil(imageWidth / (1.0 * TILE_WIDTH)), ceil(imageHeight / (1.0 * TILE_WIDTH)), 1);
  dim3 dimBlock2D(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid2D(ceil(imageWidth / (1.0 * TILE_WIDTH)), ceil(imageHeight / (1.0 * TILE_WIDTH)), 1);
  dim3 dimBlockScan(HISTOGRAM_LENGTH, 1, 1);
  dim3 dimGridScan(1, 1, 1);

  float_to_uchar_cast_kernel<<<dimGridConversion, dimBlockConversion>>>(deviceInputImage, deviceUCharImage, imageWidth, imageHeight, imageChannels);
  color_to_grayscale_kernel<<<dimGrid2D, dimBlock2D>>>(deviceUCharImage, deviceGrayScaleImage, imageWidth, imageHeight, imageChannels);
  generate_hist_kernel<<<dimGrid2D, dimBlock2D>>>(deviceGrayScaleImage, deviceHistogram, imageWidth, imageHeight);
  generate_cdf_scan_kernel<<<dimGridScan, dimBlockScan>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);
  equalize_image_kernel<<<dimGridConversion, dimBlockConversion>>>(deviceUCharImage, deviceCorrectedImage, deviceCDF, imageWidth, imageHeight, imageChannels);
  uchar_to_float_cast_kernel<<<dimGridConversion, dimBlockConversion>>>(deviceCorrectedImage, deviceOutput, imageWidth, imageHeight, imageChannels);
  
  cudaDeviceSynchronize();

  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutput, numTotalPixels * sizeof(float), cudaMemcpyDeviceToHost));

  wbSolution(args, outputImage);

  //@@ insert code here
  wbCheck(cudaFree(deviceInputImage));
  wbCheck(cudaFree(deviceUCharImage));
  wbCheck(cudaFree(deviceGrayScaleImage));
  wbCheck(cudaFree(deviceHistogram));
  wbCheck(cudaFree(deviceCorrectedImage));
  wbCheck(cudaFree(deviceCDF));
  wbCheck(cudaFree(deviceOutput));

  free(inputImage);
  free(outputImage);
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

