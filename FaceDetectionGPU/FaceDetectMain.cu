 

#include "FaceDetectExtras.h"

#include "GPUHaarCascade.h"

#include "GPU_Face_Detect.cuh"
#include "CPU_Face_Detect.h"
#include "OpenCV_Face_Detect.h"

using namespace std;
using namespace cv;


static void CheckError()
{
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("CUDA error1: %s\n", cudaGetErrorString(error));
		system("pause");
	}
}

// Loop through OpenCV Cascade to determine number of classifiers within this cascade
int numberOfClassifiers(CvHaarClassifierCascade *cvCascade)
{
	int totalClassifiers = 0;
	for(int i = 0; i < cvCascade->count; i++)
	{
		CvHaarStageClassifier stage = cvCascade->stage_classifier[i];

		totalClassifiers += stage.count;
	}

	return totalClassifiers;
}

void displayResults(IplImage * image, std::vector<CvRect> faces, char * windowTitle)
{
	// Add rectangles to demonstrate results
	for(int i = 0; i < faces.size(); i++)
	{
		CvRect face_rect = faces[i];

		cvRectangle( image, 
				cvPoint(face_rect.x, face_rect.y),
				cvPoint((face_rect.x + face_rect.width), (face_rect.y + face_rect.height)),
				CV_RGB(255, 255, 255), 3);
	}
	
	// Show results to user
	cvNamedWindow( windowTitle, 0 );
	cvShowImage( windowTitle, image );
	//cvWaitKey(0);
}

IplImage * createCopy(IplImage *img)
{
	IplImage *cpyImg = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels); 
	cvCopy(img, cpyImg);
	return cpyImg;
}

int main( int argc, const char** argv )
{
	//======================================================================================================
	// Load test image with faces to detect and OpenCVHaarCascade xml file to detect them
	//=====================================================================================================

    // Used for calculations
    int optlen = strlen("--cascade=");

    // Input file name for avi or image file.
	const char *imageFileName = "images\\lena_256.jpg";
	const char *detectorFileName;

    // Check for the correct usage of the command line
    if( argc > 1 && strncmp( argv[1], "--cascade=", optlen ) == 0 )
    {
        detectorFileName = argv[1] + optlen;
        imageFileName = argc > 2 ? argv[2] : imageFileName;
    }
    else
	{
		printf("Incorrect input for command line. Using default values instead.\n");
		printf("Correct Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n\n" );

		detectorFileName = "data\\haarcascade_frontalface_default.xml";
    }

	// Load the image
	IplImage* image;
	if((image = cvLoadImage(imageFileName, CV_LOAD_IMAGE_GRAYSCALE)) == 0)	
	{
		cout << "Error occured loading image file. Check file name?" << endl;
		system("pause");
		return 0;
	}

	int width = image->width;
	int height = image->height;
	CvSize imgSize = cvSize(width, height);

	printf("Input image: %s\n", imageFileName);
	printf("Image size: [%d, %d]\n\n", width, height);

	//======================================================================================================
	// Load OpenCVHaarCascade & create system GPUHaarCascade
	//=====================================================================================================

	CvHaarClassifierCascade *cvCascade = loadCVHaarCascade(detectorFileName);
	GPUHaarCascade gpuHaarCascade;

	// Translate OpenCV Haar Cascade data structure to GPU Haar Cascade structure
	gpuHaarCascade.load(cvCascade, numberOfClassifiers(cvCascade), imgSize); 

	printf("Input Detector: %s\n", detectorFileName);
	printf("Num of Stages: %d\n", gpuHaarCascade.numOfStages);
	printf("Num of Classifiers: %d\n\n", gpuHaarCascade.totalNumOfClassifiers);

	//======================================================================================================
	// Calculate Integral Images
	//=====================================================================================================
	
	//CvMat* sum = cvCreateMat(height + 1, width + 1, CV_32SC1);
	CvMat* sum = cvCreateMat(height + 1, width + 1, CV_32SC1);
	CvMat* sqsum = cvCreateMat(height + 1, width + 1, CV_64FC1);

	cvIntegral(image, sum, sqsum);

	//======================================================================================================
	// Calculate scale values which to resize detection window after every pass
	//=====================================================================================================
	
	double factor = 1.0f;
	float scaleFactor = 1.2f;

	std::vector<double> scale;
	
	while(factor * gpuHaarCascade.orig_window_size.width < width - 10 &&
		  factor * gpuHaarCascade.orig_window_size.height < height - 10)
	{
		scale.push_back(factor);
		factor *= scaleFactor;
	}
	
	//======================================================================================================
	// Run various Viola Jones Detection approaches
	//=====================================================================================================
	
	// Used for grouping detected rectangles
	int minNeighbors = 3;


	// Run GPU Face Detection
	initGPU(gpuHaarCascade, image, sum, sqsum);
	
	IplImage *gpuImage_v1 = createCopy(image);
	std::vector<CvRect> gpuFaces_v1 = runGPUHaarDetection(scale, minNeighbors, V1);
	
	IplImage *gpuImage_v3 = createCopy(image);
	std::vector<CvRect> gpuFaces_v3;// = runGPUHaarDetection(scale, minNeighbors, V3);
	
	IplImage *gpuImage_v4 = createCopy(image);
	std::vector<CvRect> gpuFaces_v4;// = runGPUHaarDetection(scale, minNeighbors, V4);
	
	shutDownGPU();

	// Run CPU Face Detection
	Mat sum_Mat = cvarrToMat(sum);
	Mat sqsum_Mat = cvarrToMat(sqsum);

	IplImage *cpuImage = createCopy(image);
	std::vector<CvRect> cpuFaces = runCPUHaarDetection(gpuHaarCascade, imgSize, sum_Mat, sqsum_Mat, scale, minNeighbors);

	IplImage *cpuImage_Multithread = createCopy(image);
	runCPUHaarDetection_Multithread(gpuHaarCascade, imgSize, sum_Mat, sqsum_Mat, scale, minNeighbors);

	// Run OpenCV Face Detection
	IplImage *opencvImage = createCopy(image);
	std::vector<CvRect> opencvFaces = runOpenCVHaarDetection(image, cvCascade, scaleFactor);

	// Show results
	displayResults(gpuImage_v1, gpuFaces_v1, "GPU Results v1");
	displayResults(gpuImage_v3, gpuFaces_v3, "GPU Results v3");
	displayResults(gpuImage_v4, gpuFaces_v4, "GPU Results v4");

	displayResults(cpuImage, cpuFaces, "CPU Results");
	displayResults(cpuImage_Multithread, gpuFaces_v3, "CPU_Multithread Results");

	displayResults(opencvImage, opencvFaces, "OpenCV Results");
	
	cvWaitKey(0);
	system("pause");

	//======================================================================================================
	// Free memory allocated
	//=====================================================================================================
	
	cvReleaseHaarClassifierCascade( &cvCascade );

	cvReleaseImage(&image);
	cvReleaseMat(&sum);
	cvReleaseMat(&sqsum);

	cvReleaseImage(&image);
	cvReleaseImage(&gpuImage_v1);
	cvReleaseImage(&gpuImage_v3);
	cvReleaseImage(&gpuImage_v4);
	cvReleaseImage(&cpuImage);
	cvReleaseImage(&cpuImage_Multithread);
	cvReleaseImage(&opencvImage);

	gpuHaarCascade.shutdown();

	return 0;
}


