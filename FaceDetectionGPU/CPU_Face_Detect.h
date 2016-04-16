
#include "FaceDetectExtras.h"

#include "GPUHaarCascade.h"

#ifndef CPU_FACE_DETECT_H
#define CPU_FACE_DETECT_H

std::vector<CvRect> runCPUHaarDetection(GPUHaarCascade & cascade, CvSize imgSize, cv::Mat sumImg, cv::Mat sqSumImg, std::vector<double> scale, int minNeighbors);

std::vector<CvRect> runCPUHaarDetection_Multithread(GPUHaarCascade & cascade, CvSize imgSize, cv::Mat sum, cv::Mat sqsum, std::vector<double> scale, int minNeighbors);

#endif