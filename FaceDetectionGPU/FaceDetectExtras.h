
// Standard C Library
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm> 

// OpenCV library
#include "opencv\cv.h"
#include "opencv\highgui.h"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// CUDA library
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#ifndef FACEDETECTHELPER_H
#define FACEDETECTHELPER_H

/*
struct Face
{
	int x, y;
	int width, height;
};
*/

static CvHaarClassifierCascade* loadCVHaarCascade( const char* cascade_path )
{
    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


//=========================================================================================================================
// The following code was modified to work using the OpenCV data structures & functions used throughout this implementation
//
// Reference: https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection
//=========================================================================================================================

// Threshold for grouping rectangles in post-processing stage on CPU
const float GROUP_EPS = 0.4f;

static int predicate(float eps, CvRect & r1, CvRect & r2)
{
  float delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;

  return abs(r1.x - r2.x) <= delta &&
			abs(r1.y - r2.y) <= delta &&
			abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
			abs(r1.y + r1.height - r2.y - r2.height) <= delta;
}


static int partition(std::vector<CvRect>& _vec, std::vector<int>& labels, float eps)
{
	int i, j, N = (int)_vec.size();

	CvRect* vec = &_vec[0];

	const int PARENT=0;
	const int RANK=1;

	std::vector<int> _nodes(N*2);

	int (*nodes)[2] = (int(*)[2])&_nodes[0];

	/* The first O(N) pass: create N single-vertex trees */
	for(i = 0; i < N; i++)
	{
		nodes[i][PARENT]=-1;
		nodes[i][RANK] = 0;
	}

	/* The main O(N^2) pass: merge connected components */
	for( i = 0; i < N; i++ )
	{
		int root = i;

		/* find root */
		while( nodes[root][PARENT] >= 0 )
			root = nodes[root][PARENT];

		for( j = 0; j < N; j++ )
		{
			if( i == j || !predicate(eps, vec[i], vec[j]))
			continue;
			int root2 = j;

			while( nodes[root2][PARENT] >= 0 )
				root2 = nodes[root2][PARENT];

			if( root2 != root )
			{
				/* unite both trees */
				int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
				if( rank > rank2 )
					nodes[root2][PARENT] = root;
				else
				{
					nodes[root][PARENT] = root2;
					nodes[root2][RANK] += rank == rank2;
					root = root2;
				}

				int k = j, parent;

				/* compress the path from node2 to root */
				while( (parent = nodes[k][PARENT]) >= 0 )
				{
					nodes[k][PARENT] = root;
					k = parent;
				}

				/* compress the path from node to root */
				k = i;
				while( (parent = nodes[k][PARENT]) >= 0 )
				{
					nodes[k][PARENT] = root;
					k = parent;
				}
			}
		}
	}

	/* Final O(N) pass: enumerate classes */
	labels.resize(N);
	int nclasses = 0;

	for( i = 0; i < N; i++ )
	{
		int root = i;
		while( nodes[root][PARENT] >= 0 )
			root = nodes[root][PARENT];
      
		/* re-use the rank as the class label */
		if( nodes[root][RANK] >= 0 )
			nodes[root][RANK] = ~nclasses++;

		labels[i] = ~nodes[root][RANK];
	}

	return nclasses;
}


static void groupRectangles(std::vector<CvRect> &rectList, int groupThreshold, float eps)
{
	if( groupThreshold <= 0 || rectList.empty() )
	return;

	std::vector<int> labels;

	int nclasses = partition(rectList, labels, eps);

	std::vector<CvRect> rrects(nclasses);
	std::vector<int> rweights(nclasses);

	int i, j, nlabels = (int)labels.size();

	for( i = 0; i < nlabels; i++ )
	{
		int cls = labels[i];
		rrects[cls].x += rectList[i].x;
		rrects[cls].y += rectList[i].y;
		rrects[cls].width += rectList[i].width;
		rrects[cls].height += rectList[i].height;
		rweights[cls]++;
	}
	for( i = 0; i < nclasses; i++ )
	{
		CvRect r = rrects[i];
		float s = 1.f/rweights[i];
		rrects[i].x = cvRound(r.x*s);
		rrects[i].y = cvRound(r.y*s);
		rrects[i].width = cvRound(r.width*s);
		rrects[i].height = cvRound(r.height*s);
	}

	rectList.clear();

	for( i = 0; i < nclasses; i++ )
	{
		CvRect r1 = rrects[i];
		int n1 = rweights[i];
		if( n1 <= groupThreshold )
		continue;
		/* filter out small face rectangles inside large rectangles */
		for( j = 0; j < nclasses; j++ )
		{
			int n2 = rweights[j];

			/*********************************
			* if it is the same rectangle, 
			* or the number of rectangles in class j is < group threshold, 
			* do nothing 
			********************************/
			if( j == i || n2 <= groupThreshold )
				continue;

			CvRect r2 = rrects[j];

			int dx = cvRound( r2.width * eps );
			int dy = cvRound( r2.height * eps );

			if( i != j &&
				r1.x >= r2.x - dx &&
				r1.y >= r2.y - dy &&
				r1.x + r1.width <= r2.x + r2.width + dx &&
				r1.y + r1.height <= r2.y + r2.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3) )
			break;
		}

		if( j == nclasses )
		{
		rectList.push_back(r1); // insert back r1
		}
	}
}


#endif