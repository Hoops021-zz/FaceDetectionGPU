
#include "CPU_Face_Detect.h"


using namespace std;
using namespace cv;

float calculateMean(Mat &img, CvRect rect)
{
	return img.at<int>(rect.y,rect.x) 
			- img.at<int>(rect.y,rect.x + rect.width ) 
			- img.at<int>(rect.y + rect.height , rect.x) 
			+ img.at<int>(rect.y + rect.height ,rect.x +  rect.width );
}

//float calculateSum(Mat &img, CvRect rect, int win_start_x, int win_start_y, float scale)
float calculateSum(Mat &img, GPURect rect, int win_start_x, int win_start_y, float scale)
{
	float tx = win_start_x + rect.x ;
	float ty = win_start_y + rect.y ;

	return img.at<int>(ty,tx) 
			- img.at<int>(ty,tx + rect.width ) 
			- img.at<int>(ty + rect.height , tx) 
			+ img.at<int>(ty + rect.height ,tx +  rect.width );
}
/*
float calculateSum(Mat &img, GPURect rect, int win_start_x, int win_start_y, float scale)
{
	float tx = win_start_x + rect.x * scale;
	float ty = win_start_y + rect.y * scale;

	return img.at<int>(ty,tx) 
			- img.at<int>(ty,tx + rect.width * scale) 
			- img.at<int>(ty + rect.height * scale, tx) 
			+ img.at<int>(ty + rect.height * scale,tx +  rect.width * scale);
}*/

std::vector<CvRect> haarDetection(GPUHaarCascade & gpuCascade, Mat sumImg, Mat sqSumImg)
{	
	std::vector<CvRect> detectedFaces;

	for(int i = 0; i < gpuCascade.img_detection_size.width; i++)
	{
		for(int j = 0; j < gpuCascade.img_detection_size.height; j++)
		{
			CvRect detectionWindow;
			detectionWindow.x = i;
			detectionWindow.y = j;
			detectionWindow.width = gpuCascade.real_window_size.width;
			detectionWindow.height = gpuCascade.real_window_size.height;

			float inv_window_area = 1.0f / ((float)detectionWindow.width * detectionWindow.height);
			float weightScale = inv_window_area;

			// HaarCascade file requires normalization of features
			//float mean = calculateSum(sumImg, detectionWindow, 0, 0, 1.0f) * inv_window_area;
			float mean = calculateMean(sumImg, detectionWindow);

			float variance_norm_factor = sqSumImg.at<double>(detectionWindow.y,detectionWindow.x) 
				- sqSumImg.at<double>(detectionWindow.y,detectionWindow.x + detectionWindow.width) 
				- sqSumImg.at<double>(detectionWindow.y + detectionWindow.height, detectionWindow.x) 
				+ sqSumImg.at<double>(detectionWindow.y + detectionWindow.height, detectionWindow.x + detectionWindow.width);
		
			variance_norm_factor = variance_norm_factor * inv_window_area - mean * mean;
			if(variance_norm_factor >= 0.0f)
				variance_norm_factor = sqrt(variance_norm_factor);
			else
				variance_norm_factor = 1.0f;
				
			bool passed = true;
			// for each stage
			for(int a = 0; a < gpuCascade.numOfStages; a++)
			{
				float stage_sum = 0.0;
				for(int b = 0; b < gpuCascade.haar_stage_classifiers[a].numofClassifiers; b++)
				{
					//int classifier_offset = stageClassifiers[i].classifierOffset;
					int index = b +  gpuCascade.haar_stage_classifiers[a].classifierOffset;
					//GPUHaarClassifier classifier = gpuCascade.haar_classifiers[index];
					GPUHaarClassifier classifier = gpuCascade.scaled_haar_classifiers[index];

					double t = classifier.threshold * variance_norm_factor;

					double sum = calculateSum(sumImg, classifier.haar_feature.rect0.r, detectionWindow.x, detectionWindow.y, gpuCascade.scale) * classifier.haar_feature.rect0.weight * weightScale;
					sum += calculateSum(sumImg, classifier.haar_feature.rect1.r, detectionWindow.x, detectionWindow.y, gpuCascade.scale) * classifier.haar_feature.rect1.weight * weightScale;

					// If there is a third rect
					if(classifier.haar_feature.rect2.weight)
						sum += calculateSum(sumImg, classifier.haar_feature.rect2.r, detectionWindow.x, detectionWindow.y, gpuCascade.scale) * classifier.haar_feature.rect2.weight * weightScale;
            
					if(sum >= t)
						stage_sum += classifier.alpha1;
					else
						stage_sum += classifier.alpha0;
				}

				// Classifier did not pass, abort entire cascade
				if( stage_sum < gpuCascade.haar_stage_classifiers[a].threshold)
				{
					passed = false;
					break;
				}
			}
	
			// Face detected
			if(passed)
				detectedFaces.push_back(detectionWindow);
		}
	}

	return detectedFaces;
}

std::vector<CvRect> runCPUHaarDetection(GPUHaarCascade &cascade ,CvSize imgSize, Mat sumImg, Mat sqSumImg, std::vector<double> scale, int minNeighbors)
{
	printf("****Beginning CPU Haar Detection****\n\n");

	clock_t startCPU, endCPU;
	
	GPUHaarCascade gpuCascade;	
	gpuCascade.load(&cascade);

	std::vector<CvRect> allDetectedFaces;

	double totalElapsedTime = 0.0f;
	for(int i = 0; i < scale.size(); i++)
	{
		gpuCascade.setFeaturesForScale(scale[i]);

		startCPU = clock();

		std::vector<CvRect> faces = haarDetection(gpuCascade, sumImg, sqSumImg);

		endCPU = clock();
		double elapsedTime = (double)endCPU - startCPU;
		totalElapsedTime += elapsedTime;

		// Output performance results
		printf("Stage: %d // Faces Detected: %d // GPU Time: %3.1f ms \n", i, faces.size(), elapsedTime);

		// Append detected faces for this scaled window into main array
		allDetectedFaces.insert(allDetectedFaces.end(), faces.begin(), faces.end());
	}

	// Output final performance
	printf("\nTotal compute time: %3.1f ms \n\n", totalElapsedTime);

	// Group detected faces for cleaner results
	if( minNeighbors != 0)
	{
		groupRectangles(allDetectedFaces, minNeighbors, GROUP_EPS);
	}

	// Clean up the cascade
	gpuCascade.shutdown();

	return allDetectedFaces;
}