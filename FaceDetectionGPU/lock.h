/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __LOCK_H__
#define __LOCK_H__

#include "cuda_runtime.h"

struct Lock 
{
    int *mutex;
	/*
    Lock( void ) 
	{
        HANDLE_ERROR( cudaMalloc( (void**)&mutex, sizeof(int) ) );
        HANDLE_ERROR( cudaMemset( mutex, 0, sizeof(int) ) );
    }

    ~Lock( void ) {
        cudaFree( mutex );
    }*/

	void init()
	{
		HANDLE_ERROR( cudaMalloc( (void**)&mutex, sizeof(int) ) );
        HANDLE_ERROR( cudaMemset( mutex, 0, sizeof(int) ) );
	}

    __device__ void lock( void ) {
		#if __CUDA_ARCH__ >= 110 
			while( atomicCAS( mutex, 0, 1 ) != 0 );
		#endif
    }

    __device__ void unlock( void ) {
		#if __CUDA_ARCH__ >= 110 
			atomicExch( mutex, 0 );
		#endif
    }

	void shutdown()
	{
		cudaFree( mutex );
	}
};

#endif
