/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * als.cu
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */
//do not use magma and fp16 by default  
//#define CUMF_USE_MAGMA
//#define CUMF_USE_HALF
#include "als.h"
#include "host_utilities.h"
#include <fstream>
#include <assert.h>
#include <cuda_fp16.h> 
#ifdef CUMF_USE_HALF
#define SCAN_BATCH 24
#else
#define SCAN_BATCH 24
#endif
#ifdef CUMF_USE_MAGMA
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#endif

__global__ void fp32Array2fp16Array(const float * fp32Array, half* fp16Array,
		const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		fp16Array[i] =  __float2half(fp32Array[i]);
	}
}

int updateX(const int batch_size, const int batch_offset, float * ythetaT, float * tt, float * XT,
		cublasHandle_t handle, const int m, const int n, const int f, const int nnz,
		float** devPtrTTHost, float **devPtrYthetaTHost){
	//variables for timing
	float elapsed;
	struct timeval tv0, tv1, tv2;
	gettimeofday(&tv0, NULL);
	printf("*******Batch LU factorization of tt.\n");
	//pointers needed by batch op
	float **devPtrTT = 0;
	int *INFO;
	for (int k = 0; k < batch_size; k++) {
		devPtrTTHost[k] = &tt[k * f * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrTT, batch_size * sizeof(*devPtrTT)));
	cudacall(cudaMemcpy(devPtrTT, devPtrTTHost, batch_size * sizeof(*devPtrTT),cudaMemcpyHostToDevice));
	//cudacall( cudaMalloc(&P, f * batch_size * sizeof(int)) );
	cudacall( cudaMalloc(&INFO, batch_size * sizeof(int) ));
	cublascall(cublasSgetrfBatched(handle, f, devPtrTT, f, NULL, INFO, batch_size));

	cudaThreadSynchronize();
	gettimeofday(&tv1, NULL);
	elapsed = (tv1.tv_sec - tv0.tv_sec)
			+ (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);

	printf("*******solve: tt * XT = ythetaT use cublas, with LU decomposition.\n");

	float **devPtrYthetaT = 0;

	for (int k = 0; k < batch_size; k++) {
		devPtrYthetaTHost[k] = &ythetaT[batch_offset * f + k * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrYthetaT, batch_size * sizeof(*devPtrYthetaT)));
	cudacall(cudaMemcpy(devPtrYthetaT, devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaT), cudaMemcpyHostToDevice));

	int * info2 = (int *) malloc(sizeof(int));
	cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
			(const float ** ) devPtrTT, f, NULL, devPtrYthetaT, f, info2, batch_size) );

	cudaThreadSynchronize();
	cudaError_t cudaStat1 = cudaGetLastError();
	if (cudaStat1 != cudaSuccess) {
		fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
		exit(EXIT_FAILURE);
	}

	cudacall( cudaMemcpy(&XT[batch_offset * f], &ythetaT[batch_offset * f],
			batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice) );
	gettimeofday(&tv2, NULL);
	elapsed = (tv2.tv_sec - tv1.tv_sec)
			+ (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);

	cudacall(cudaFree(devPtrTT));
	//cudacall(cudaFree(P));
	cudacall(cudaFree(INFO));
	cudacall(cudaFree(devPtrYthetaT));
	return 0;
}

int updateTheta(const int batch_size, const int batch_offset, float * xx,
		  float * yTXT, float * thetaT,
		cublasHandle_t handle,
		 const int m, const int n, const int f, const int nnz,
		 float ** devPtrXXHost, float **devPtrYTXTHost ){

	//variables for timing
	float elapsed;
	struct timeval tv0, tv1, tv2;

	gettimeofday(&tv0, NULL);
	printf("*******LU factorize xx.\n");
	float **devPtrXX = 0;

	for (int k = 0; k < batch_size; k++) {
		devPtrXXHost[k] = &xx[k * f * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrXX, batch_size * sizeof(*devPtrXX)));
	cudacall(cudaMemcpy(devPtrXX, devPtrXXHost, batch_size * sizeof(*devPtrXX), cudaMemcpyHostToDevice));
	int *INFO;
	//cudacall(cudaMalloc(&P, f * batch_size * sizeof(int)));
	cudacall(cudaMalloc(&INFO, batch_size * sizeof(int)));
	cublascall(cublasSgetrfBatched(handle, f, devPtrXX, f, NULL, INFO, batch_size));
	cudaThreadSynchronize();

	gettimeofday(&tv1, NULL);
	elapsed = (tv1.tv_sec - tv0.tv_sec)
			+ (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);

	printf("******* solve xx * thetaT = yTXT with CUDA 7.\n");

	float **devPtrYTXT = 0;

	for (int k = 0; k < batch_size; k++) {
		devPtrYTXTHost[k] = &yTXT[batch_offset * f + k * f];
	}

	cudacall(cudaMalloc((void** ) &devPtrYTXT, batch_size * sizeof(*devPtrYTXT)));
	cudacall(cudaMemcpy(devPtrYTXT, devPtrYTXTHost, batch_size * sizeof(*devPtrYTXT),cudaMemcpyHostToDevice));

	int * info2 = (int *) malloc(sizeof(int));
	cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
			(const float ** ) devPtrXX, f, NULL, devPtrYTXT, f, info2, batch_size) );
	cudaThreadSynchronize();
	cudaError_t cudaStat1 = cudaGetLastError();
	if (cudaStat1 != cudaSuccess) {
		fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
		exit(EXIT_FAILURE);
	}

	cudacall( cudaMemcpy( &thetaT[batch_offset * f], &yTXT[batch_offset * f],
	                        batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice) );
	gettimeofday(&tv2, NULL);
	elapsed = (tv2.tv_sec - tv1.tv_sec)
			+ (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);

	cudaFree(devPtrXX);
	cudaFree(INFO);
	free(info2);
	cudaFree(devPtrYTXT);
	return 0;
}

#ifdef USE_MAGMA
int updateThetaMagma(const int batch_size, const int batch_offset, float * xx,
		  float * yTXT, float * thetaT,
		cublasHandle_t handle,
		 const int m, const int n, const int f, const int nnz,
		 float ** devPtrXXHost, float **devPtrYTXTHost ){
	//variables for timing
	float elapsed;
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	printf("*******magma Cholesky factorization.\n");
	magma_init();
	magma_opts opts( MagmaOptsBatched );
	char *parray[10];
	char **x;
	x = &parray[0];
	opts.parse_opts(1,x);
	magma_queue_t queue = opts.queue;
	int min_batch = batch_size;
	int info = 0;
	int * dinfo_array = 0;
	float **dA_array = NULL; 
	float **dB_array = NULL;
	float **hA_array = (float**) malloc(min_batch * sizeof(hA_array[0]));
	float **hB_array = (float**) malloc(min_batch * sizeof(hB_array[0]));	

	cudacall (cudaMalloc((void**) &dinfo_array, min_batch*sizeof(int)));
	cudacall(cudaMalloc((void** ) &dA_array, min_batch * sizeof(*dA_array)));
	cudacall(cudaMalloc((void** ) &dB_array, min_batch * sizeof(*dB_array)));
	for (int k = 0; k < batch_size; k++) {
		hA_array[k] = &xx[k * f * f];
		hB_array[k] = &yTXT[batch_offset * f + k * f];
	}
	cudacall(cudaMemcpy(dA_array, hA_array, min_batch * sizeof(*dA_array), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(dB_array, hB_array, min_batch * sizeof(*dB_array), cudaMemcpyHostToDevice));

	info = magma_sposv_batched(MagmaLower, f, 1, dA_array, f, dB_array, f, dinfo_array, min_batch, queue);
	magma_int_t *dipiv;
	magma_int_t     **dipiv_array = NULL;
	TESTING_MALLOC_DEV( dipiv, magma_int_t, f * min_batch );
	TESTING_MALLOC_DEV( dipiv_array, magma_int_t*,     min_batch );
	magma_iset_pointer( dipiv_array, dipiv, 1, 0, 0, f, min_batch, queue );

	//info = magma_sgesv_nopiv_batched(f, 1, dA_array, f, dB_array, f, dinfo_array, min_batch, queue);
    //info = magma_sgesv_batched(f, 1, dA_array, f, dipiv_array, dB_array, f, dinfo_array, min_batch, queue);

	int *cpu_info = (int*) malloc(min_batch*sizeof(int));
	cudacall(cudaMemcpy(cpu_info, dinfo_array, min_batch * sizeof(int),cudaMemcpyDeviceToHost));

	cudacall( cudaMemcpy(&thetaT[batch_offset * f], &yTXT[batch_offset * f],
			batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	for(int i = 0; i < min_batch; i++){
		if(cpu_info[i] != 0 ){
			printf("magma_sposv_batched matrix %d returned internal error %d\n",i, (int)cpu_info[i] );
		}
	}
	if (info != 0)
		printf("magma_sposv_batched returned argument error %d: %s.\n", (int) info, magma_strerror( info ));
	
	cudaFree(dA_array);
	cudaFree(dB_array);
	cudaFree( dinfo_array );
	cudaFree(dipiv_array);
	cudaFree(dipiv);
	free(cpu_info);
	free(hA_array);
	free(hB_array);
	//free(x);
	magma_finalize();
	gettimeofday(&tv2, NULL);
	elapsed = (tv2.tv_sec - tv1.tv_sec)
			+ (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);
	return 0;
}
#endif

__global__ void RMSE(const float * csrVal, const int* cooRowIndex,
		const int* csrColIndex, const float * __restrict__ thetaT, const float * __restrict__ XT, float * error, const int nnz,
		const int error_size, const int f, const float avg_rating) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nnz) {
		int row = cooRowIndex[i];
		int col = csrColIndex[i];
		float e = csrVal[i] - avg_rating;
		//if(i%1000000==0) printf("row: %d, col: %d, csrVal[%d]: %f.\t", row, col, i, e);
		for (int k = 0; k < f; k++) {
			e -= __ldg(&thetaT[f * col + k]) * __ldg(&XT[f * row + k]);
		}
		atomicAdd(&error[i%error_size], e*e);
		//error[i] = e*e;
		//if(i%1000000==0) printf("error[%d]: %f.\n", i, e);
	}
}
__global__ void RMSE(const float * csrVal, const int* cooRowIndex,
		const int* csrColIndex, const float * __restrict__ thetaT, const float * __restrict__ XT, float * error, const int nnz,
		const int error_size, const int f, const float avg_rating, const float * user_bias, const float * item_bias) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nnz) {
		int row = cooRowIndex[i];
		int col = csrColIndex[i];
		float e = csrVal[i] - avg_rating;
		//if(i%1000000==0) printf("row: %d, col: %d, csrVal[%d]: %f.\t", row, col, i, e);
		for (int k = 0; k < f; k++) {
			e -= __ldg(&thetaT[f * col + k]) * __ldg(&XT[f * row + k]);
		}
		e -= user_bias[row] + item_bias[col];
		atomicAdd(&error[i%error_size], e*e);
		//error[i] = e*e;
		//if(i%1000000==0) printf("error[%d]: %f.\n", i, e);
	}
}

//using fp16 as thetaT's format
//using fp16 in computate seems causing register pressure since half intrinsics cannot be used.
//using fp16 in compute also does not converge. not sure if the code is incorrect, or ALS cannot tolerate half-precision
__global__ void
__launch_bounds__(64, 6)
get_hermitian100WithHalf(const int batch_offset, float* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const half* __restrict__ thetaT_fp16) {
	extern __shared__ float2 thetaTemp[];
	int row = blockIdx.x + batch_offset;
	if (row < m) {
		//this block needs to handle end - start thetaT columns
		int start = csrRowIndex[row];
		int end = csrRowIndex[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1)/SCAN_BATCH + 1;
		
		float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
		float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
		float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
		float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
		float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
		float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
		float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
		float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
		float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
		float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;
	
		int tile_x = 0;
		int tile_y = 0;

		int tile = F/10;
		for ( int i = 0; i < 10; i++){
			int end = ((20-i)*(i+1))/2;
			if(threadIdx.x < end){
				tile_x = i * tile;
				tile_y = (10 + threadIdx.x - end) * tile;
				break;
			}
		}
		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter ++){
			//float2 theta;
			//copy texture --> smem, and sync
			//two layers: warp divergence unless we split at 32
			//require: 32 >= SCAN_BATCH
			if(threadIdx.x < 2*32 ){
				int index = threadIdx.x - (threadIdx.x/32)*32;	//0 to 31;
				if(index < SCAN_BATCH){
					if(iter*SCAN_BATCH + index < end - start){
						//for (int k = 50*(threadIdx.x/32); k < 50*(threadIdx.x/32) + 50; k += 2){
						//IMPORTANT: for loop has constant and identical start and end
						if(threadIdx.x < 32){
							for (int k = 0; k < 50; k += 2){
								half2 theta_half2 = __ldg((half2*)&thetaT_fp16[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k]);
								thetaTemp[index * F/2 + k/2] = __half22float2(theta_half2);
								//theta.x = __half2float(__ldg(&thetaT_fp16[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k]));
								//theta.y = __half2float(__ldg(&thetaT_fp16[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k+1]));
								//thetaTemp[index * F/2 + k/2] = theta;
							}
						}
						else {
							for (int k = 0; k < 50; k += 2){
								half2 theta_half2 = __ldg((half2*)&thetaT_fp16[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 50]);
								thetaTemp[index * F/2 + k/2 + 25] = __half22float2(theta_half2);
								//theta.x = __half2float(__ldg(&thetaT_fp16[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 50]));
								//theta.y = __half2float(__ldg(&thetaT_fp16[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 51]));
								//thetaTemp[index * F/2 + k/2 + 25] = theta;
							}
						}
					}
					//must be the last iteration; no need to check
					//not enough theta to copy, set zero
					else
						memset(&thetaTemp[index*F/2], 0, F*sizeof(float));
				}
			}
			__syncthreads();
			//tile: 10*10
			if(threadIdx.x < 55 ){
				for(int k = 0; k < SCAN_BATCH; k++){
					accumulate_in_registers();
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();

		if(threadIdx.x < 55 ){
			//copy output to gmem
			int index = blockIdx.x*F*F;
			fill_lower_half_from_registers();
			//symmetric
			if(tile_x!=tile_y){
				fill_upper_half_from_registers();
			}
			//regularization
			if(tile_x == tile_y){
				for(int k = 0; k < tile; k++)
					tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}
	}
}

__global__ void
__launch_bounds__(64, 6)
get_hermitian100(const int batch_offset, float* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const float* __restrict__ thetaT) {
	extern __shared__ float2 thetaTemp[];
	int row = blockIdx.x + batch_offset;
	if (row < m) {
		//this block needs to handle end - start thetaT columns
		int start = csrRowIndex[row];
		int end = csrRowIndex[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1)/SCAN_BATCH + 1;
		float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
		float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
		float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
		float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
		float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
		float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
		float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
		float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
		float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
		float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

		int tile_x = 0;
		int tile_y = 0;

		int tile = F/10;
		for ( int i = 0; i < 10; i++){
			int end = ((20-i)*(i+1))/2;
			if(threadIdx.x < end){
				tile_x = i * tile;
				tile_y = (10 + threadIdx.x - end) * tile;
				break;
			}
		}
		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter ++){
			float2 theta;
			//copy texture --> smem, and sync
/*
			if(threadIdx.x < SCAN_BATCH){
				if(iter*SCAN_BATCH + threadIdx.x < end - start){
					for (int k = 0; k < F; k += 2){
						theta.x = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + threadIdx.x] + k);
						theta.y = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + threadIdx.x] + k+1);
						thetaTemp[threadIdx.x * F/2 + k/2] = theta;
					}
				}
				//must be the last iteration; no need to check
				//not enough theta to copy, set zero
				else
					memset(&thetaTemp[threadIdx.x*F/2], 0, F*sizeof(float));
			}
*/

			//two layers: warp divergence unless we split at 32
			//require 32 >= SCAN_BATCH
			if(threadIdx.x < 2*32 ){
				//int index = threadIdx.x;
				int index = threadIdx.x - (threadIdx.x/32)*32;	//0 to 31;
				if(index < SCAN_BATCH){
					if(iter*SCAN_BATCH + index < end - start){
						//for (int k = 50*(threadIdx.x/32); k < 50*(threadIdx.x/32) + 50; k += 2){
						//IMPORTANT: for loop has constant and identical start and end
						if(threadIdx.x < 32){
							for (int k = 0; k < 50; k += 2){
								theta.x = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k]);
								theta.y = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k+1]);
								thetaTemp[index * F/2 + k/2] = theta;
							}
						}
						else {
							for (int k = 0; k < 50; k += 2){
								theta.x = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 50]);
								theta.y = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 51]);
								thetaTemp[index * F/2 + k/2 + 25] = theta;
							}
						}
					}
					//must be the last iteration; no need to check
					//not enough theta to copy, set zero
					else
						memset(&thetaTemp[index*F/2], 0, F*sizeof(float));
				}
			}


/*			//issue: not coalesced access to csrColIndex
			if(threadIdx.x < F && threadIdx.x%2 == 0){
				for(int k = 0; k< SCAN_BATCH; k++){
					if(iter*SCAN_BATCH + k < end - start){
						theta.x = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + k] + threadIdx.x);
						theta.y = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + k] + threadIdx.x +1);
						thetaTemp[k * F/2 + threadIdx.x/2] = theta;
					}
					//must be the last iteration; no need to check
					//not enough theta to copy, set zero
					else
						memset(&thetaTemp[k*F/2 + threadIdx.x/2], 0, 2*sizeof(float));
				}
			}
*/
/*
			int layers = blockDim.x/SCAN_BATCH;	//100/30 = 3
			//int height = blockDim.x/layers; //30
			int y = threadIdx.x/SCAN_BATCH;//0,1,2,3; y==3 is not viable
			//min(y, (layers-1)) * height
			int y_start = y * 30;//0-29:0;30-59:30;60-89:60
			int y_end = y_start + 30;	//0-29:30;30-59:60;60-89:90
			if(y >= layers - 1) y_end = F;	//60-89:100
			if(threadIdx.x - y_start < SCAN_BATCH){
				if(iter*SCAN_BATCH + (threadIdx.x - y_start) < end - start){
					for (int k = y_start; k < y_end; k += 2){
						theta.x =
								tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + (threadIdx.x - y_start)] + k);
						theta.y =
								tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + (threadIdx.x - y_start)] + k+1);
						thetaTemp[(threadIdx.x - y_start)* F/2 + k/2] = theta;
					}
				}
				//must be the last iteration; no need to check
				//not enough theta to copy, set zero
				else
					memset(&thetaTemp[(threadIdx.x - y_start)*F/2 + y_start/2], 0, (y_end - y_start)*sizeof(float));
			}

*/
			__syncthreads();

			//tile: 10*10
			if(threadIdx.x < 55 ){
				for(int k = 0; k < SCAN_BATCH; k++){
					accumulate_in_registers();
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();

		if(threadIdx.x < 55 ){
			//copy output to gmem
			int index = blockIdx.x*F*F;
			fill_lower_half_from_registers();
			//symmetric
			if(tile_x!=tile_y){
				fill_upper_half_from_registers();
			}
			//regularization
			if(tile_x == tile_y){
				for(int k = 0; k < tile; k++)
					tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}
	}
}

/*a generic kernel to get the hermitian matrices
 * as the left-hand side of the equations, to update X in ALS
 *examplary F = 100, T = 10
 */
__global__ void
get_hermitianT10(const int batch_offset, float* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const float* __restrict__ thetaT) {
	extern __shared__ float2 thetaTemp [];
	int row = blockIdx.x + batch_offset;
	if (row < m) {
		//this block needs to handle end - start thetaT columns
		int start = csrRowIndex[row];
		int end = csrRowIndex[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1)/SCAN_BATCH + 1;
		float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
		float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
		float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
		float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
		float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
		float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
		float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
		float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
		float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
		float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

		int N = F/T10; // N = 100/10=10; for F = 100 and T = 10
		int effective_block_size = N*(N+1)/2;
		//get the x and y coordinate
		int tile_x = 0;
		int tile_y = 0;
		for ( int i = 0; i < N; i++ ) {
			int end = ((2*N-i)*(i+1))/2;
			if(threadIdx.x < end){
				tile_x = i * T10;
				tile_y = (N + threadIdx.x - end) * T10;
				break;
			}
		}
		int index = blockIdx.x*F*F;
		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter ++){
			//phase 1 in iteration: gmem --> smem
			//REQ: blockDim.x >= F/2
			if(threadIdx.x < F/2){
				for(int k = 0; k< SCAN_BATCH; k++){
					if(iter*SCAN_BATCH + k < end - start){
						float2 theta;
						theta.x = __ldg(&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x]);
						theta.y = __ldg(&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x+1]);
						thetaTemp[k * F/2 + threadIdx.x] = theta;
						//this simpler statement is slower.
						//thetaTemp[k * F/2 + threadIdx.x] = __ldg((float2*)&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x]);
					}
					//not enough theta to copy, set zero
					else
						memset(&thetaTemp[k*F/2 + threadIdx.x], 0, 2*sizeof(float));
				}
			}
			__syncthreads();
			
			//phase 2 in iteration: smem --> register
			if(threadIdx.x < effective_block_size){//this redundant "if" seems improving kernel performance
				for(int k = 0; k < SCAN_BATCH; k++){
					accumulate_in_registers();
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();

		//phase 3, after iteration: register --> gmem
		if(threadIdx.x < effective_block_size){
			fill_lower_half_from_registers();

			//symmetric
			if(tile_x != tile_y){
				fill_upper_half_from_registers();
			}
			//regularization
			if(tile_x == tile_y){
				for(int k = 0; k < T10; k++)
					tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}
	}
}

__global__ void
normalize_csr_ratings(float* csrVal, const int* csrRowIndex, const int* csrColIndex, 
	const float* user_bias, const float* item_bias, const float avg_rating,
	const int m){
	//block per user row
	if (blockIdx.x < m) {		
		int start = csrRowIndex[blockIdx.x];
		int end = csrRowIndex[blockIdx.x + 1];
		for(int index = threadIdx.x; index < end - start; index += blockDim.x){
			csrVal[start + index] -=
				user_bias[blockIdx.x] + item_bias[csrColIndex[start + index]] + avg_rating;
		}
	}
}

__global__ void
normalize_csc_ratings(float* cscVal, const int* cscRowIndex, const int* cscColIndex, 
	const float* user_bias, const float* item_bias, const float avg_rating,
	const int n){
	//block per item col
	if (blockIdx.x < n) {		
		int start = cscColIndex[blockIdx.x];
		int end = cscColIndex[blockIdx.x + 1];
		for(int index = threadIdx.x; index < end - start; index += blockDim.x){
			cscVal[start + index] -=
				item_bias[blockIdx.x] + user_bias[cscRowIndex[start + index]] + avg_rating;
		}
	}
}


__global__ void
normalize_csr_errors(float* csrVal, const int* csrRowIndex, const int* csrColIndex, 
	const float* user_bias, const float* item_bias, const float avg_rating,
	const int m, const int F, const float* thetaT, const float * xT, const float lambda){
	//block per user row
	if (blockIdx.x < m) {		
		int start = csrRowIndex[blockIdx.x];
		int end = csrRowIndex[blockIdx.x + 1];
		for(int index = threadIdx.x; index < end - start; index += blockDim.x){
			float temp = item_bias[csrColIndex[start + index]] + avg_rating;
			for(int k = 0; k < F; k++)
				temp += thetaT[ F * csrColIndex[start + index] + k] * xT[F * blockIdx.x + k];
			csrVal[start + index] = (csrVal[start + index] - temp)/(1 + lambda)/(end - start); 
		}
	}
}

__global__ void
normalize_csc_errors(float* cscVal, const int* cscRowIndex, const int* cscColIndex, 
	const float* user_bias, const float* item_bias, const float avg_rating,
	const int n, const int F, const float* thetaT, const float * xT, const float lambda){
	//block per user row
	if (blockIdx.x < n) {		
		int start = cscColIndex[blockIdx.x];
		int end = cscColIndex[blockIdx.x + 1];
		for(int index = threadIdx.x; index < end - start; index += blockDim.x){
			float temp = user_bias[cscRowIndex[start + index]] + avg_rating;
			for(int k = 0; k < F; k++)
				temp += thetaT[ F * blockIdx.x + k] * xT[F * cscRowIndex[start + index] + k];
			//cscVal[start + index] -= temp; 
			cscVal[start + index] = (cscVal[start + index] - temp)/(1 + lambda)/(end - start); 
		}
	}
}


void doALS(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const float* csrValHostPtr,
		const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const float* cscValHostPtr,
		const int* cooRowIndexHostPtr, float* thetaTHost, float* XTHost,
		const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const float * cooValHostTestPtr,
		const int m, const int n, const int f, const long nnz, const long nnz_test, const float lambda,
		const int ITERS, const int X_BATCH, const int THETA_BATCH, const float avg_rating)
{
	//device pointers
	int * csrRowIndex = 0;
	int * csrColIndex = 0;
	float * csrVal = 0;
	float * thetaT = 0;
	float * tt = 0;
	float * XT = 0;
	float * cscVal =0;
	int * cscRowIndex = 0;
	int * cscColIndex = 0;
	//coo to calculate RMSE
	int * cooRowIndex =0;
	float * cooVal_test;
	int * cooRowIndex_test;
	int * cooColIndex_test;
	//bias terms
	float * user_bias;
	float * item_bias;
	
	float* ones_m_host;
	float* ones_n_host;
	cudacall(cudaMallocHost( (void** ) &ones_m_host, m * sizeof(ones_m_host[0])) );
	cudacall(cudaMallocHost( (void** ) &ones_n_host, n * sizeof(ones_n_host[0])) );
	float* ones_m;
	float* ones_n;
	cudacall(cudaMalloc( (void** ) &ones_m, m * sizeof(ones_m[0])) );
	cudacall(cudaMalloc( (void** ) &ones_n, n * sizeof(ones_n[0])) );

	for (int k = 0; k < n; k++)
		ones_n_host[k] = 1.0f;
	for (int k = 0; k < m; k++)
		ones_m_host[k] = 1.0f;
	cudacall(cudaMemcpy(ones_m, ones_m_host,(size_t ) m * sizeof(ones_m[0]), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(ones_n, ones_n_host,(size_t ) n * sizeof(ones_n[0]), cudaMemcpyHostToDevice));

	printf("*******start allocating memory on GPU...\n");
	cudacall(cudaMalloc((void** ) &cscRowIndex,nnz * sizeof(cscRowIndex[0])));
	cudacall(cudaMalloc((void** ) &cscColIndex, (n+1) * sizeof(cscColIndex[0])));
	cudacall(cudaMalloc((void** ) &cscVal, nnz * sizeof(cscVal[0])));
	//dimension: F*N
	cudacall(cudaMalloc((void** ) &thetaT, f * n * sizeof(thetaT[0])));
	//dimension: M*F
	cudacall(cudaMalloc((void** ) &XT, f * m * sizeof(XT[0])));
	
	cudacall(cudaMalloc((void** ) &user_bias, m * sizeof(user_bias[0])));
	cudacall(cudaMalloc((void** ) &item_bias, n * sizeof(item_bias[0])));
	cudacall(cudaMemset(user_bias, 0, m * sizeof(user_bias[0])));
	cudacall(cudaMemset(item_bias, 0, n * sizeof(item_bias[0])));

	printf("*******start copying memory to GPU...\n");

	cudacall(cudaMemcpy(cscRowIndex, cscRowIndexHostPtr,(size_t ) nnz * sizeof(cscRowIndex[0]), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(cscColIndex, cscColIndexHostPtr,(size_t ) (n+1) * sizeof(cscColIndex[0]), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(cscVal, cscValHostPtr,(size_t ) (nnz * sizeof(cscVal[0])),cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(thetaT, thetaTHost, (size_t ) (n * f * sizeof(thetaT[0])), cudaMemcpyHostToDevice));

	cudacall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	//64-bit smem access
	//http://acceleware.com/blog/maximizing-shared-memory-bandwidth-nvidia-kepler-gpus
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	//initialize cublas, cusparse
	cublasHandle_t handle;
	cublascall(cublasCreate(&handle));
	cusparseHandle_t cushandle = 0;
	cusparsecall(cusparseCreate(&cushandle));
	cusparseMatDescr_t descr;
	cusparsecall( cusparseCreateMatDescr(&descr));
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	using namespace std;
	//variable used to time
	double elapsed = 0.0;
	struct timeval tv;
	struct timeval start_tv;
	struct timeval start_tv2;

	for(int iter = 0; iter < ITERS ; iter ++){
		printf("---------------------------ALS iteration %d, update X.----------------------------------\n", iter);
		gettimeofday(&start_tv, NULL);
		//copy csr matrix in
		cudacall(cudaMalloc((void** ) &csrRowIndex,(m + 1) * sizeof(csrRowIndex[0])));
		cudacall(cudaMalloc((void** ) &csrColIndex, nnz * sizeof(csrColIndex[0])));
		cudacall(cudaMalloc((void** ) &csrVal, nnz * sizeof(csrVal[0])));
		cudacall(cudaMemcpy(csrRowIndex, csrRowIndexHostPtr,(size_t ) ((m + 1) * sizeof(csrRowIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(csrColIndex, csrColIndexHostPtr,(size_t ) (nnz * sizeof(csrColIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(csrVal, csrValHostPtr,(size_t ) (nnz * sizeof(csrVal[0])),cudaMemcpyHostToDevice));

		printf("\tgenerate: Y*theta using cusparse.\n");
		float * ytheta = 0;
		float * ythetaT = 0;
		cudacall(cudaMalloc((void** ) &ytheta, f * m * sizeof(ytheta[0])));
		cudacall(cudaMalloc((void** ) &ythetaT, f * m * sizeof(ythetaT[0])));

		//normalize csrVal with bias terms
		normalize_csr_ratings<<<m, 64>>>
			(csrVal, csrRowIndex, csrColIndex, user_bias, item_bias, avg_rating, m);
		cudaDeviceSynchronize();
		cudaCheckError();

		const float alpha = 1.0f;
		const float beta = 0.0f;
		cusparsecall (cusparseScsrmm2(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, m, f, n, nnz, &alpha, descr, csrVal,
				csrRowIndex, csrColIndex, thetaT, f, &beta, ytheta, m) );
		//cudaDeviceSynchronize();
		//printf("*******transpose ytheta use cublas.\n");
		//ytheta: m*f; need ythetaT = (ytheta).T = f*m
		cublascall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, f, m, &alpha,
				(const float * ) ytheta, m, &beta, ythetaT, f, ythetaT, f));
		cudaDeviceSynchronize();
		cudaCheckError();
		cudacall(cudaFree(ytheta));
		cudacall(cudaFree(csrVal));
		gettimeofday(&tv, NULL);
		elapsed = (tv.tv_sec - start_tv.tv_sec)
				+ (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
		printf("\tgenerate: Y*theta run %f seconds.\n", elapsed);

		int block_dim = f/T10*(f/T10+1)/2;
		if (block_dim < f/2) block_dim = f/2;
		for(int batch_id = 0; batch_id< X_BATCH; batch_id ++){
			printf("*******batch %d / %d.*******\n", batch_id, X_BATCH);
			int batch_size = 0;
			if(batch_id != X_BATCH - 1)
				batch_size = m/X_BATCH;
			else
				batch_size = m - batch_id*(m/X_BATCH);
			int batch_offset = batch_id * (m/X_BATCH);
			cudacall(cudaMalloc((void** ) &tt, f * f * batch_size * sizeof(float)));
			gettimeofday(&start_tv2, NULL);
			printf("\tupdateXByBlock kernel.\n");
			if(f == 100){
				//do not use fp16 by default
				#ifdef CUMF_USE_HALF
				half* thetaT_fp16 = 0;
				cudacall(cudaMalloc((void** ) &thetaT_fp16, f * n * sizeof(thetaT_fp16[0])));
				fp32Array2fp16Array<<<(n*f-1)/1024 + 1, 1024>>>(thetaT, thetaT_fp16, f*n);
				get_hermitian100WithHalf<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2)>>>
					(batch_offset, tt, csrRowIndex, csrColIndex, lambda, m, f, thetaT_fp16);
				cudacall(cudaFree(thetaT_fp16));
				#else
				get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2)>>>
					(batch_offset, tt, csrRowIndex, csrColIndex, lambda, m, f, thetaT);
				#endif
			}
			else
				get_hermitianT10<<<batch_size, block_dim, SCAN_BATCH * f/2*sizeof(float2)>>>
					(batch_offset, tt, csrRowIndex, csrColIndex, lambda, m, f, thetaT);
			cudaDeviceSynchronize();
			cudaCheckError();
			gettimeofday(&tv, NULL);
			elapsed = (tv.tv_sec - start_tv2.tv_sec)
					+ (tv.tv_usec - start_tv2.tv_usec) / 1000000.0;
			printf("\tupdate X kernel run %f seconds, gridSize: %d, blockSize %d.\n", elapsed, batch_size, f);

			//host pointers for cublas batch operations
			double t0 = seconds();
			float ** devPtrTTHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrTTHost, batch_size * sizeof(*devPtrTTHost) ) );
			float **devPtrYthetaTHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaTHost) ) );

			printf("\tinvoke updateX with batch_size: %d, batch_offset: %d..\n", batch_size, batch_offset);
			updateX(batch_size, batch_offset, ythetaT, tt, XT, handle, m, n, f, nnz,
					devPtrTTHost, devPtrYthetaTHost);

			printf("\tupdateX run seconds: %f \n", seconds() - t0);
			cudacall(cudaFree(tt));
			cudacall(cudaFreeHost(devPtrTTHost));
			cudacall(cudaFreeHost(devPtrYthetaTHost));
		}
		gettimeofday(&tv, NULL);
		elapsed = (tv.tv_sec - start_tv.tv_sec)
				+ (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
		printf("update X run %f seconds, gridSize: %d, blockSize %d.\n", elapsed, m, f);
		cudacall(cudaFree(csrRowIndex));
		cudacall(cudaFree(csrColIndex));
		cudacall(cudaFree(ythetaT));

		gettimeofday(&start_tv, NULL);
		printf("---------------------------------- ALS iteration %d, update theta ----------------------------------\n", iter);
		
		//normalize cscVal with bias terms

		cudacall(cudaMemcpy(cscVal, cscValHostPtr,(size_t ) (nnz * sizeof(cscVal[0])),cudaMemcpyHostToDevice));
		normalize_csc_ratings<<<n,64>>>
			(cscVal, cscRowIndex, cscColIndex, user_bias, item_bias, avg_rating, n);
		cudaDeviceSynchronize();
		cudaCheckError();
		
		printf("\tgenerate: Y'*X using cusparse.\n");
		
		float * yTX = 0;
		float * yTXT = 0;
		cudacall(cudaMalloc((void** ) &yTXT, f * n * sizeof(yTXT[0])));
		cudacall(cudaMalloc((void** ) &yTX, n * f * sizeof(yTX[0])));
		cusparsecall( cusparseScsrmm2(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, n, f, m, nnz, &alpha, descr, cscVal,
				cscColIndex, cscRowIndex, XT, f, &beta, yTX, n) );
		//cudaDeviceSynchronize();
		//printf("*******transpose yTX \n");
		//yTX: n*f; need yTXT = (yTX).T = f*n
		cublascall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, f, n, &alpha,
				(const float * ) yTX, n, &beta, yTXT, f, yTXT, f));
		cudaDeviceSynchronize();
		cudacall(cudaFree(yTX));
		gettimeofday(&tv, NULL);
		elapsed = (tv.tv_sec - start_tv.tv_sec)
				+ (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
		printf("\tgenerate: Y'*X run %f seconds.\n", elapsed);
		//in batches, when N is huge
		for(int batch_id = 0; batch_id< THETA_BATCH; batch_id ++){
			printf("*******batch %d / %d.*******\n", batch_id, THETA_BATCH);
			int batch_size = 0;
			if(batch_id != THETA_BATCH - 1)
				batch_size = n/THETA_BATCH;
			else
				batch_size = n - batch_id*(n/THETA_BATCH);
			int batch_offset = batch_id * (n/THETA_BATCH);

			float * xx = 0;
			cudacall(cudaMalloc((void** ) &xx, f * f * batch_size * sizeof(xx[0])));
			cudacall( cudaMemset(xx, 0, f*f*batch_size*sizeof(float)) );

			gettimeofday(&start_tv2, NULL);
			printf("\tupdateThetaByBlock kernel.\n");
			//get_hermitian_theta<<<batch_size, 64>>>(batch_offset, xx, cscRowIndex, cscColIndex, lambda, n);
			//updateThetaByBlock2pRegDsmemTile<<<batch_size, F>>>
			if(f == 100){
				#ifdef CUMF_USE_HALF
				half * XT_fp16 = 0;
				cudacall(cudaMalloc((void** ) &XT_fp16, f * m * sizeof(XT_fp16[0])));
				fp32Array2fp16Array<<<(n*f-1)/1024 + 1, 1024>>>(XT, XT_fp16, f*m);
				get_hermitian100WithHalf<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2)>>>
					(batch_offset, xx, cscColIndex, cscRowIndex, lambda, n, f, XT_fp16);
				cudacall(cudaFree(XT_fp16));
				#else
				get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2)>>>
					(batch_offset, xx, cscColIndex, cscRowIndex, lambda, n, f, XT);
				#endif
			}
			else
				get_hermitianT10<<<batch_size, block_dim, SCAN_BATCH*f*sizeof(float)>>>
					(batch_offset, xx, cscColIndex, cscRowIndex, lambda, n, f, XT);
			cudaDeviceSynchronize();
			cudaCheckError();

			gettimeofday(&tv, NULL);
			elapsed = (tv.tv_sec - start_tv2.tv_sec)
					+ (tv.tv_usec - start_tv2.tv_usec) / 1000000.0;
			printf("\tupdate Theta kernel run %f seconds, gridSize: %d, blockSize %d.\n",
					elapsed, batch_size, f);

			double t0 = seconds();
			float ** devPtrXXHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrXXHost, batch_size * sizeof(*devPtrXXHost) ) );
			float **devPtrYTXTHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrYTXTHost, batch_size * sizeof(*devPtrYTXTHost) ) );
			printf("*******invoke updateTheta with batch_size: %d, batch_offset: %d.\n", batch_size, batch_offset);
			updateTheta(batch_size, batch_offset, xx, yTXT, thetaT, handle, m,  n,  f,  nnz,
					devPtrXXHost, devPtrYTXTHost);
			printf("\tupdateTheta run seconds: %f \n", seconds() - t0);
			cudacall(cudaFree(xx));
			cudacall(cudaFreeHost(devPtrXXHost));
			cudacall(cudaFreeHost(devPtrYTXTHost));
		}
		cudacall(cudaFree(yTXT));
		gettimeofday(&tv, NULL);
		elapsed = (tv.tv_sec - start_tv.tv_sec)
				+ (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
		printf("update theta run %f seconds, gridSize: %d, blockSize %d.\n",
				elapsed, n, f);

		printf("update bias terms.\n");
		cudacall(cudaMalloc((void** ) &csrVal, nnz * sizeof(csrVal[0])));
		cudacall(cudaMemcpy(csrVal, csrValHostPtr,(size_t ) (nnz * sizeof(csrVal[0])),cudaMemcpyHostToDevice));
		cudacall(cudaMalloc((void** ) &csrRowIndex, (m+1) * sizeof(csrRowIndex[0])));
		cudacall(cudaMemcpy(csrRowIndex, csrRowIndexHostPtr,(size_t ) ((m+1) * sizeof(csrRowIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMalloc((void** ) &csrColIndex, nnz * sizeof(csrColIndex[0])));
		cudacall(cudaMemcpy(csrColIndex, csrColIndexHostPtr,(size_t ) (nnz * sizeof(csrColIndex[0])), cudaMemcpyHostToDevice));
		normalize_csr_errors<<<m, 32>>>
			(csrVal, csrRowIndex, csrColIndex, user_bias, item_bias, avg_rating, m, f, thetaT, XT, lambda);
		cudaDeviceSynchronize();
		cudaCheckError();

		//load original cscVal
		cudacall(cudaMemcpy(cscVal, cscValHostPtr,(size_t ) (nnz * sizeof(cscVal[0])),cudaMemcpyHostToDevice));
		normalize_csc_errors<<<n, 32>>>
			(cscVal, cscRowIndex, cscColIndex, user_bias, item_bias, avg_rating, n, f, thetaT, XT, lambda);
		cudaDeviceSynchronize();
		cudaCheckError();
		//update user_bias <- csr*ones_m
		cusparsecall (cusparseScsrmv(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			m, n, nnz, &alpha, descr, csrVal, csrRowIndex, csrColIndex, ones_n, &beta, user_bias) );
		cudaDeviceSynchronize();
		cudaCheckError();

		//update item_bias <- csc*ones_n
		cusparsecall (cusparseScsrmv(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, m, nnz, &alpha, descr, cscVal, cscColIndex, cscRowIndex, ones_m, &beta, item_bias) );
		cudaDeviceSynchronize();
		cudaCheckError();
		
		cudacall(cudaFree(csrRowIndex));
		cudacall(cudaFree(csrColIndex));
		cudacall(cudaFree(csrVal));
		printf("Calculate RMSE.\n");
		float * errors_train = 0;
		int error_size = 1000;
		cudacall(cudaMalloc((void** ) &errors_train, error_size * sizeof(errors_train[0])));
		cudacall( cudaMemset(errors_train, 0, error_size*sizeof(float)) );

		cudacall(cudaMalloc((void** ) &cooRowIndex, nnz * sizeof(cooRowIndex[0])));
		cudacall(cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,(size_t ) (nnz * sizeof(cooRowIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMalloc((void** ) &csrColIndex, nnz * sizeof(csrColIndex[0])));
		cudacall(cudaMalloc((void** ) &csrVal, nnz * sizeof(csrVal[0])));
		cudacall(cudaMemcpy(csrColIndex, csrColIndexHostPtr,(size_t ) (nnz * sizeof(csrColIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(csrVal, csrValHostPtr,(size_t ) (nnz * sizeof(csrVal[0])),cudaMemcpyHostToDevice));

		RMSE<<<(nnz-1)/256 + 1, 256>>>
				(csrVal, cooRowIndex, csrColIndex, thetaT, XT, errors_train, nnz, error_size, f, avg_rating, user_bias, item_bias);
		cudaDeviceSynchronize();
		cudaCheckError();
		cudacall(cudaFree(cooRowIndex));
		cudacall(cudaFree(csrColIndex));
		cudacall(cudaFree(csrVal));

		float* rmse_train = (float*) malloc (sizeof(float));
		cublascall( cublasSasum(handle, error_size, errors_train, 1, rmse_train) );

		cudaDeviceSynchronize();
		printf("@@@@@@@@@@@@@@@@@@@ Train RMSE in iter %d: %f\n", iter, sqrt((*rmse_train)/nnz));
		cudacall(cudaFree(errors_train));

		
		float * errors_test = 0;
		cudacall(cudaMalloc((void** ) &errors_test, error_size * sizeof(errors_test[0])));
		cudacall( cudaMemset(errors_test, 0, error_size*sizeof(float)) );

		cudacall(cudaMalloc((void** ) &cooRowIndex_test, nnz_test * sizeof(cooRowIndex_test[0])));
		cudacall(cudaMemcpy(cooRowIndex_test, cooRowIndexTestHostPtr,(size_t ) (nnz_test * sizeof(cooRowIndex_test[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMalloc((void** ) &cooColIndex_test, nnz_test * sizeof(cooColIndex_test[0])));
		cudacall(cudaMalloc((void** ) &cooVal_test, nnz_test * sizeof(cooVal_test[0])));
		cudacall(cudaMemcpy(cooColIndex_test, cooColIndexTestHostPtr,(size_t ) (nnz_test * sizeof(cooColIndex_test[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(cooVal_test, cooValHostTestPtr,(size_t ) (nnz_test * sizeof(cooVal_test[0])),cudaMemcpyHostToDevice));

		RMSE<<<(nnz_test-1)/256, 256>>>(cooVal_test, cooRowIndex_test, cooColIndex_test, thetaT, XT,
				errors_test, nnz_test, error_size, f, avg_rating, user_bias, item_bias);
		cudaDeviceSynchronize();
		cudaCheckError();

		cudacall(cudaFree(cooRowIndex_test));
		cudacall(cudaFree(cooColIndex_test));
		cudacall(cudaFree(cooVal_test));

		float* rmse_test = (float*) malloc (sizeof(float));
		cublascall( cublasSasum(handle, error_size, errors_test, 1, rmse_test) );
		cudaDeviceSynchronize();
		printf("@@@@@@@@@@@@@@@@@@@ Test RMSE in iter %d: %f\n", iter, sqrt((*rmse_test)/nnz_test));
		cudacall(cudaFree(errors_test));
		
	}
	/*
	//save model to a file
	cudacall(cudaMemcpy(ones_m_host, user_bias, (size_t ) (m * sizeof(user_bias[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaMemcpy(ones_n_host, item_bias, (size_t ) (n * sizeof(item_bias[0])), cudaMemcpyDeviceToHost));
	FILE * user_file = fopen("user.bias", "wb");
	FILE * item_file = fopen("item.bias", "wb");
	fwrite(ones_m_host, sizeof(float), m, user_file);
	fwrite(ones_n_host, sizeof(float), n, item_file);
	fclose(user_file);
	fclose(item_file);
	*/

	
	//copy feature vectors back to host
	cudacall(cudaMemcpy(thetaTHost, thetaT, (size_t ) (n * f * sizeof(thetaT[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaMemcpy(XTHost, XT, (size_t ) (m * f * sizeof(XT[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaFree(thetaT));
	cudacall(cudaFree(XT));
	cudacall(cudaFree(cscVal));
	cudacall(cudaFree(cscColIndex));
	cudacall(cudaFree(cscRowIndex));
	cudacall(cudaDeviceReset());
}
