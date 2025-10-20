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
#define BLOCK_DIM 256

//@@ insert code here
__global__ void float_to_uchar_cast_kernel(float *input_image, unsigned char *output_image, int image_width, int image_height, int image_channels) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  if (col < image_width && row < image_height) {
    int index = (row * image_width + col) * image_channels + channel;
    output_image[index] =  (unsigned char)(255 * input_image[index]);
  }
}

__global__ void uchar_to_float_cast_kernel(unsigned char *input_image, float *output_image, int image_width, int image_height, int image_channels) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  if (col < image_width && row < image_height) {
    int index = (row * image_width + col) * image_channels + channel;
    output_image[index] =  (float)(input_image[index] / 255.0);
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

__global__ void correct_image_kernel(unsigned char *image, unsigned char *corrected_image, float *cdf, int image_width, int image_height, int imageChannels) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  if (col < image_width && row < image_height) {
    int index = (row * image_width + col) * imageChannels + channel;
    corrected_image[index] = (unsigned char)(apply_cdf_correction(image[index], cdf, cdf[0]));
  }
}

void calculate_cdf(unsigned int *histogram, float *cdf, int image_width, int image_height) {
  cdf[0] = ((float) histogram[0]) / (image_width * image_height);
  for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
    cdf[i] = cdf[i - 1] + (((float) histogram[i]) / (image_width * image_height));
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

  unsigned int *hostHistogram;
  float *hostCDF;

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

  //@@ TODO: delete
  hostHistogram = (unsigned int *) malloc(sizeof(unsigned int) * HISTOGRAM_LENGTH);
  hostCDF = (float *) malloc(sizeof(unsigned int) * HISTOGRAM_LENGTH);

  wbCheck(cudaMalloc((void **)&deviceInputImage, numTotalPixels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceUCharImage, numTotalPixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrayScaleImage, numPixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCorrectedImage, numTotalPixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numTotalPixels * sizeof(float)));

  wbCheck(cudaMemcpy(deviceInputImage, hostInputImageData, numTotalPixels * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));

  dim3 dimBlockConversion(TILE_WIDTH, TILE_WIDTH, imageChannels);
  dim3 dimGridConversion(ceil(imageWidth / (1.0 * TILE_WIDTH)), ceil(imageHeight / (1.0 * TILE_WIDTH)), 1);
  dim3 dimBlock2D(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid2D(ceil(imageWidth / (1.0 * TILE_WIDTH)), ceil(imageHeight / (1.0 * TILE_WIDTH)), 1);

  float_to_uchar_cast_kernel<<<dimGridConversion, dimBlockConversion>>>(deviceInputImage, deviceUCharImage, imageWidth, imageHeight, imageChannels);
  color_to_grayscale_kernel<<<dimGrid2D, dimBlock2D>>>(deviceUCharImage, deviceGrayScaleImage, imageWidth, imageHeight, imageChannels);
  generate_hist_kernel<<<dimGrid2D, dimBlock2D>>>(deviceGrayScaleImage, deviceHistogram, imageWidth, imageHeight);
  
  cudaDeviceSynchronize();

  //TODO: make this a kernel
  wbCheck(cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  calculate_cdf(hostHistogram, hostCDF, imageWidth, imageHeight);
  wbCheck(cudaMemcpy(deviceCDF, hostCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
  
  correct_image_kernel<<<dimGridConversion, dimBlockConversion>>>(deviceUCharImage, deviceCorrectedImage, deviceCDF, imageWidth, imageHeight, imageChannels);
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
  free(hostCDF);
  free(hostHistogram);

  return 0;
}

