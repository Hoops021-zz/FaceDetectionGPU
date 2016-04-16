
#include "FaceDetectExtras.h"

#include "GPUHaarCascade.h"

#ifndef FACE_DETECT_CUH
#define FACE_DETECT_CUH

#include "lock.h"

//======================================================================================================
// Define kernels that can be launched
//=====================================================================================================
enum FaceDetectionKernel { V1, V2, V3, V4};

float launchKernel_v1(int width, int height);
float launchKernel_v2(int width, int height);
float launchKernel_v3(int width, int height);
float launchKernel_v4(int width, int height);

__global__ void haarDetection_v1(GPUHaarCascade haarCascade, GPURect * detectedFaces);
__global__ void haarDetection_v2(GPUHaarCascade haarCascade, GPURect * detectedFaces);
__global__ void haarDetection_v3(GPUHaarCascade haarCascade, GPURect * detectedFaces);
__global__ void haarDetection_v4(GPUHaarCascade haarCascade, GPURect * detectedFaces, Lock * locks);

#define THREADS_PER_BLOCK_V4 64
#define WARP_SIZE 32
#define DETECTION_WINDOW_STRIDE_V4 THREADS_PER_BLOCK_V4/WARP_SIZE

// GPU Detection Engine
void initGPU(GPUHaarCascade &h_gpuHaarCascade, IplImage * image, CvMat *sumImg, CvMat *sqSumImg);
std::vector<CvRect> runGPUHaarDetection(std::vector<double> scale, int minNeighbors, FaceDetectionKernel kernelSelect);
void shutDownGPU();

// Kernel Init methods
void allocateGPUCascade(GPUHaarCascade &h_gpuCascade, GPUHaarCascade &dev_gpuCascade);
void allocateIntegralImagesGPU(CvMat * sumImage, CvMat *sqSumImage, cudaArray *dev_sumArray, cudaArray * dev_sqSumArray);
void releaseTextures();



#endif





