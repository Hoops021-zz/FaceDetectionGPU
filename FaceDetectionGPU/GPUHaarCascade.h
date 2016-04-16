
#include "opencv\cv.h"

#ifndef GPUHAARCASCADE_H
#define GPUHAARCASCADE_H

/*
 * OpenCV HaarCascade is tree based and nodes point to host memory address. 
 * Following data structures provide means to flatten tree into
 * 1D array in memory with index offset values stored for each stage_classifier 
 * instead of pointers to accomodate easier memory transfer to the GPU device
 */

// NOTE: This structure assumes haar trees are stump-based to simplify implementation


typedef struct GPURect
{
	float x;
	float y;
	float width;
	float height;
#ifdef __cplusplus
	__device__ __host__
	GPURect(float _x = 0, float _y = 0, float w = 0, float h = 0) : x(_x), y(_y), width(w), height(h) {}
#endif
}GPURect;



typedef struct GPUFeatureRect
{
	GPURect r;
	float weight;
}GPUFeatureRect;

typedef struct GPUHaarFeature
{
	GPUFeatureRect rect0;
	GPUFeatureRect rect1;
	GPUFeatureRect rect2;
}GPUHaarFeature;

typedef struct GPUHaarClassifier
{
	// For this implementaiton, assume only one feature per classifier
	GPUHaarFeature haar_feature;
	float threshold;

	// corresponds to alpha[]
	float alpha0;
	float alpha1;
}GPUHaarClassifier;

typedef struct GPUHaarStageClassifier
{
	//number of classifiers in this stage
	int  numofClassifiers; 

	//  threshold for the boosted classifier
    float threshold; 
    
	// Index offset pointing to beginning of Classifiers for this stage
	int classifierOffset;
}GPUHaarStageClassifier;

typedef struct GPUHaarCascade
{
	// CvHaarClassifierCascade Parameters
	int  flags; /* signature */
    int  numOfStages; /* number of stages */
	int totalNumOfClassifiers; /* total number of classifiers in cascade */
    CvSize orig_window_size; /* original object size (the cascade is trained for) */
	CvSize img_window_size; /* size of original window */

	CvSize img_detection_size; /* set at run of haar detection */

    /* these two parameters are set by cvSetImagesForHaarClassifierCascade */
    CvSize real_window_size; /* current object size */
    double scale; /* current scale */

	// Array of all stage classifiers in system
    GPUHaarStageClassifier* haar_stage_classifiers; 

	// Array of all classifiers in cascade
	GPUHaarClassifier * haar_classifiers;

	// Arrray of all classifiers modified with the above scale value
	GPUHaarClassifier * scaled_haar_classifiers;
	
	void load(CvHaarClassifierCascade *cvCascade, int clasifiersCount, CvSize imgSize)
	{
		// Transfer basic cascade properties over
		flags = cvCascade->flags;
		numOfStages = cvCascade->count;
		orig_window_size = cvCascade->orig_window_size;
		real_window_size = cvCascade->real_window_size;
		img_window_size = imgSize;
		scale = cvCascade->scale;
		totalNumOfClassifiers = clasifiersCount;

		haar_stage_classifiers = (GPUHaarStageClassifier *)malloc(numOfStages * sizeof(GPUHaarStageClassifier));
		haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));
		scaled_haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));

		// Loop through OpenCV Cascade tree to transfer Stage_Classifiers & Classifiers data
		int gpuClassifierCounter = 0;
		for(int i = 0; i < cvCascade->count; i++)
		{
			// Grab OpenCV stage classifier
			CvHaarStageClassifier stage = cvCascade->stage_classifier[i];
		
			// Create GPU Stage Classifier
			GPUHaarStageClassifier gpuStage;
			gpuStage.threshold = stage.threshold;
			gpuStage.numofClassifiers = stage.count;
			gpuStage.classifierOffset = gpuClassifierCounter;

			// Loop through all classifiers for this stage
			for(int j = 0; j < stage.count; j++)
			{
				CvHaarClassifier classifier = stage.classifier[j];

				if(classifier.count > 1)
				{
					// TODO: throw error
					printf("Can't handle HaarFeature xml files with classifiers with more than 1 haar feature.\n");
					return;
				}

				// Create GPU Classifier
				GPUHaarClassifier gpuClassifier;
				gpuClassifier.threshold = classifier.threshold[0];
				//gpuClassifier.left = classifier.left[0];
				//gpuClassifier.right = classifier.right[0];
				gpuClassifier.alpha0 = classifier.alpha[0];
				gpuClassifier.alpha1 = classifier.alpha[1];

				// Grab OpenCV Haar Feature
				CvHaarFeature feature = classifier.haar_feature[0];

				// Create GPU feature
				GPUHaarFeature gpuFeature;
				//gpuFeature.rect0.r = feature.rect[0].r;
				gpuFeature.rect0.r.x = feature.rect[0].r.x;
				gpuFeature.rect0.r.y = feature.rect[0].r.y;
				gpuFeature.rect0.r.width = feature.rect[0].r.width;
				gpuFeature.rect0.r.height = feature.rect[0].r.height;

				gpuFeature.rect0.weight = feature.rect[0].weight;

				//gpuFeature.rect1.r = convertRect(feature.rect[1].r);
				gpuFeature.rect1.r.x = feature.rect[1].r.x;
				gpuFeature.rect1.r.y = feature.rect[1].r.y;
				gpuFeature.rect1.r.width = feature.rect[1].r.width;
				gpuFeature.rect1.r.height = feature.rect[1].r.height;
				gpuFeature.rect1.weight = feature.rect[1].weight;

				//gpuFeature.rect2.r = convertRect(feature.rect[2].r);
				gpuFeature.rect2.r.x = feature.rect[2].r.x;
				gpuFeature.rect2.r.y = feature.rect[2].r.y;
				gpuFeature.rect2.r.width = feature.rect[2].r.width;
				gpuFeature.rect2.r.height = feature.rect[2].r.height;
				gpuFeature.rect2.weight = feature.rect[2].weight;

				gpuClassifier.haar_feature = gpuFeature;

				// Add new GPU Classifier to array of classifiers in GPUCascade
				haar_classifiers[gpuClassifierCounter] = gpuClassifier;
				scaled_haar_classifiers[gpuClassifierCounter] = gpuClassifier;
				gpuClassifierCounter++;
			}

			// Add new GPU stage classifier to array
			haar_stage_classifiers[i] = gpuStage;
		}
	}

	void load(GPUHaarCascade * gpuCascade)
	{
		// Transfer basic cascade properties over
		flags = gpuCascade->flags;
		numOfStages = gpuCascade->numOfStages;
		orig_window_size = gpuCascade->orig_window_size;
		real_window_size = gpuCascade->real_window_size;
		img_window_size = gpuCascade->img_window_size;
		scale = gpuCascade->scale;
		totalNumOfClassifiers = gpuCascade->totalNumOfClassifiers;

		// Allocate & copy stage clasifiers
		haar_stage_classifiers = (GPUHaarStageClassifier *)malloc(numOfStages * sizeof(GPUHaarStageClassifier));
		for(int i = 0; i < numOfStages; i++)
			haar_stage_classifiers[i] = gpuCascade->haar_stage_classifiers[i];

		// Allocate & copy classifiers
		haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));
		scaled_haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));
		for(int i = 0; i < totalNumOfClassifiers; i++)
		{
			haar_classifiers[i] = gpuCascade->haar_classifiers[i];
			scaled_haar_classifiers[i] = gpuCascade->scaled_haar_classifiers[i];
		}
	}

	void setFeaturesForScale(float newScale)
	{
		scale = newScale;
		real_window_size.width = cvRound(orig_window_size.width * scale);
		real_window_size.height = cvRound(orig_window_size.height * scale);

		img_detection_size.width = cvRound(img_window_size.width - real_window_size.width);
		img_detection_size.height = cvRound(img_window_size.height - real_window_size.height);

		for(int i = 0; i < totalNumOfClassifiers; i++)
		{
			GPUHaarFeature original_feature = haar_classifiers[i].haar_feature;
			GPUHaarFeature *scaled_feature = &scaled_haar_classifiers[i].haar_feature;

			scaled_feature->rect0.r.x = original_feature.rect0.r.x * scale;
			scaled_feature->rect0.r.y = original_feature.rect0.r.y * scale;
			scaled_feature->rect0.r.width = original_feature.rect0.r.width * scale;
			scaled_feature->rect0.r.height = original_feature.rect0.r.height * scale;
			
			scaled_feature->rect1.r.x = original_feature.rect1.r.x * scale;
			scaled_feature->rect1.r.y = original_feature.rect1.r.y * scale;
			scaled_feature->rect1.r.width = original_feature.rect1.r.width * scale;
			scaled_feature->rect1.r.height = original_feature.rect1.r.height * scale;
			
			if(original_feature.rect2.weight)
			{
				scaled_feature->rect2.r.x = original_feature.rect2.r.x * scale;
				scaled_feature->rect2.r.y = original_feature.rect2.r.y * scale;
				scaled_feature->rect2.r.width = original_feature.rect2.r.width * scale;
				scaled_feature->rect2.r.height = original_feature.rect2.r.height * scale;
			}
		}
	}

	void shutdown()
	{
		free(haar_stage_classifiers);
		free(haar_classifiers);
	}

}GPUHaarCascade;

#endif