

#ifndef OPENCV_FACE_DETECT_H
#define OPENCV_FACE_DETECT_H

#include "FaceDetectExtras.h"

std::vector<CvRect> runOpenCVHaarDetection(IplImage *image, CvHaarClassifierCascade* cascade, float scaleFactor);

#endif