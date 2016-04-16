
#include "OpenCV_Face_Detect.h"

using namespace std;
using namespace cv;

CvHaarClassifierCascade* load_object_detector( const char* cascade_path )
{
    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}

std::vector<CvRect> detectObjects( IplImage* image, CvHaarClassifierCascade* cascade, int do_pyramids, float scaleFactor )
{
    IplImage* small_image = image;
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* faces;

    /* if the flag is specified, down-scale the input image to get a
       performance boost w/o loosing quality (perhaps) */
    if( do_pyramids )
    {
        small_image = cvCreateImage( cvSize(image->width/2,image->height/2), IPL_DEPTH_8U, 3 );
        cvPyrDown( image, small_image, CV_GAUSSIAN_5x5 );
        scaleFactor = 2;
    }

    /* use the fastest variant */
    faces = cvHaarDetectObjects( small_image, cascade, storage, scaleFactor, 2, CV_HAAR_DO_CANNY_PRUNING );

	std::vector<CvRect> returnFaces;	
	for(int i = 0; i < faces->total; i++ )
	{
		// Extract the rectangles only
		CvRect face_rect = *(CvRect*)cvGetSeqElem( faces, i );

		returnFaces.push_back(face_rect);
	}

    if( small_image != image )
        cvReleaseImage( &small_image );
    cvReleaseMemStorage( &storage );

	return returnFaces;
}

std::vector<CvRect> runOpenCVHaarDetection(IplImage *image, CvHaarClassifierCascade* cascade, float scaleFactor)
{
	printf("****Beginning OpenCV Haar Detection****\n\n");
	clock_t start, end;

	std::vector<CvRect> outputFaces;

	start = clock();

		outputFaces = detectObjects( image, cascade, 0, scaleFactor);

	end = clock();
	double elapsedTime = (double)end - start;
	printf("\nTotal compute time: %3.1f ms \n\n", elapsedTime);

	return outputFaces;
}

/*
IplImage *im_gray = cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
cvCvtColor(image, im_gray, CV_RGB2GRAY);
*/

/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// REFERENCE: http://www.cs264.org/2009/projects/web/Dai_Yi/Hsiao_Dai_Website/Project_Write_Up.html


int main( int argc, char** argv )
{
	
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;


}*/