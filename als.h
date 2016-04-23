/*
 * als.h
 *
 *  Created on: Aug 13, 2015
 *      Author: weitan
 */

#ifndef ALS_H_
#define ALS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <fstream>
#include <cusparse.h>
#include <host_defines.h>
//these parameters do not change among different problem size
//our kernels handle the case where F%T==0 and F = 100
#define T10 10

#define cudacall(call) \
    do\
    {\
	cudaError_t err = (call);\
	if(cudaSuccess != err)\
	    {\
		fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	    }\
    }\
    while (0)\

#define cublascall(call) \
do\
{\
	cublasStatus_t status = (call);\
	if(CUBLAS_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cusparsecall(call) \
do\
{\
	cusparseStatus_t status = (call);\
	if(CUSPARSE_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUSPARSE Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }\
while(0)\

inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

extern "C" {

void loadCSRSparseMatrixBin(const char* dataFile, const char* rowFile, const char* colFile,
		float* data, int* row, int* col, const int m, const long nnz);

void loadCSCSparseMatrixBin(const char* dataFile, const char* rowFile, const char* colFile,
		float * data, int* row, int* col, const int n, const long nnz);

void loadCooSparseMatrixRowPtrBin(const char* rowFile, int* row, const long nnz);

void loadCooSparseMatrixBin(const char* dataFile, const char* rowFile, const char* colFile,
		float* data, int* row, int* col, const long nnz);

}

void doALS(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const float* csrValHostPtr,
		const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const float* cscValHostPtr,
		const int* cooRowIndexHostPtr, float* thetaTHost, float * XTHost,
		const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const float * cooValHostTestPtr,
		const int m, const int n, const int f, const long nnz, const long nnz_test, const float lambda,
		const int ITERS, const int X_BATCH, const int THETA_BATCH);

#endif /* ALS_H_ */
