/*
 * hugewiki.cu
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <cusparse.h>
#include <host_defines.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <sstream>
#include "./common.h"
#include "../als.h"
#include "../cg.h"
//variable definition
#define F 100
#define TILE_SIZE F/10
#define SCAN_BATCH 30
#define THETA_BATCH 3
#define X_BATCH 240
#define ITERS 10
#define M  50082603
#define N 39780
#define NNZ 3101144313
#define NNZ_TEST 344573330
//0.05 when use both "full" kernels
#define LAMBDA 0.048

//hardware specific
#define GPU_COUNT 4
#define DEVICEID 0 // the anchor device

//debug option to save model
//#define CUMF_SAVE_MODEL
//#define CUMF_TT_FP16


using namespace std;
void saveDeviceFloatArrayToFile(string fileName, int size, float* d_array){
	float* h_array;
	cudacall(cudaMallocHost( (void** ) &h_array, size * sizeof(h_array[0])) );
	cudacall(cudaMemcpy(h_array, d_array, size * sizeof(h_array[0]),cudaMemcpyDeviceToHost));
	FILE * outfile = fopen(fileName.c_str(), "wb");
	fwrite(h_array, sizeof(float), size, outfile);
	fclose(outfile);
	cudaFreeHost(h_array);
}

__global__ void
__launch_bounds__(64, 6)
get_hermitian100_tt_fp16(const int batch_offset, half2* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m,
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
		#ifdef DEBUG
		//if(threadIdx.x==0)
		//	printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
		#endif
		if(threadIdx.x < 55 ){
			//weighted-lambda regularization
			if(tile_x == tile_y){
				float temp = (end - start) * lambda;
				temp0 += temp;
				temp11 += temp;
				temp22 += temp;
				temp33 += temp;
				temp44 += temp;
				temp55 += temp;
				temp66 += temp;
				temp77 += temp;
				temp88 += temp;
				temp99 += temp;
			}
			//copy output to gmem
			int index = blockIdx.x*F*F/2;
			//fill_lower_half_from_registers();
			fill_lower_half_from_registers_fp16();
			//symmetric
			if(tile_x!=tile_y){
				//fill_upper_half_from_registers();
				fill_upper_half_from_registers_fp16();
			}
		}
	}
}

__global__ void
__launch_bounds__(64, 6) 
get_hermitian100(const int batch_offset, float* tt, const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const float* __restrict__ thetaT) {
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
		#ifdef DEBUG
		//if(threadIdx.x==0)
		//	printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
		#endif
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

//split a big csr into many by rows. the row id of sub-matrices need to be changed
//inval = inval - inval[0]
__global__ void zeroIndex(int * inVal, const unsigned int inVal_0, const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < size){
		inVal[i] = (unsigned)inVal[i] - inVal_0;
	}
}

texture<float> xTTexRef;
texture<float> thetaTTexRef;

__global__ void
__launch_bounds__(100, 4)
updateThetaByBlock2pRegDsmemTile(float * xx, const int* cscRowIndex,
		const int* cscColIndex, const float lambda, const float * XT) {
	__shared__ float2 xTemp[SCAN_BATCH * F/2];
		int col = blockIdx.x;
		if (col < N) {
			//this block needs to handle end - start XT columns
			int start = cscColIndex[col];
			int end = cscColIndex[col + 1];

			int iterations = (end - start -1)/SCAN_BATCH + 1;
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
			float2 x;
			int tile = F/10;
			int tile_x = (threadIdx.x/tile) * tile;//start x of this tile
			int tile_y = (threadIdx.x%tile) * tile;//start y of this tile

			for (int iter = 0; iter < iterations; iter ++){
				//copy texture --> smem, and sync
				if(threadIdx.x < SCAN_BATCH){
					if(iter*SCAN_BATCH + threadIdx.x < end - start){
						for (int k = 0; k < F; k += 2){
							x.x =
									XT[ F * cscRowIndex[start + iter*SCAN_BATCH + threadIdx.x] + k ];
							x.y =
									XT [ F * cscRowIndex[start + iter*SCAN_BATCH + threadIdx.x] + k+1 ];
							xTemp[threadIdx.x * F/2 + k/2] = x;
						}
					}
					//must be the last iteration; no need to check
					//not enough theta to copy, set zero
					else
						memset(&xTemp[threadIdx.x*F/2], 0, F*sizeof(float));
				}
				__syncthreads();
				///////////////////////////////////////////////////////////////////////////////////////////////////////////
				//tile: 10*10
				for(int k = 0; k < SCAN_BATCH; k++){
					temp0 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp1 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp2 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp3 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp4 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp5 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp6 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp7 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp8 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp9 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp10 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp11 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp12 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp13 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp14 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp15 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp16 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp17 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp18 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp19 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;

					temp20 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp21 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp22 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp23 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp24 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp25 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp26 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp27 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp28 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp29 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp30 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp31 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp32 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp33 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp34 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp35 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp36 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp37 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp38 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp39 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;

					temp40 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp41 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp42 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp43 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp44 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp45 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp46 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp47 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp48 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp49 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp50 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp51 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp52 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp53 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp54 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp55 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp56 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp57 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp58 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp59 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;

					temp60 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp61 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp62 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp63 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp64 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp65 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp66 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp67 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp68 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp69 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp70 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp71 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp72 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp73 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp74 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp75 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp76 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp77 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp78 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp79 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;


					temp80 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp81 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp82 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp83 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp84 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp85 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp86 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp87 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp88 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp89 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp90 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp91 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp92 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp93 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp94 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp95 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp96 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp97 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp98 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp99 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;
				}
				///////////////////////////////////////////////////////////////////////////////////////////////////////////
				__syncthreads();
			}

			int index = blockIdx.x*F*F;
	///*
			//copy output to gmem
			xx[index + tile_x + tile_y*F] = temp0;
			xx[index + tile_x + (tile_y + 1)*F] = temp1;
			xx[index + tile_x + (tile_y + 2)*F] = temp2;
			xx[index + tile_x + (tile_y + 3)*F] = temp3;
			xx[index + tile_x + (tile_y + 4)*F] = temp4;
			xx[index + tile_x + (tile_y + 5)*F] = temp5;
			xx[index + tile_x + (tile_y + 6)*F] = temp6;
			xx[index + tile_x + (tile_y + 7)*F] = temp7;
			xx[index + tile_x + (tile_y + 8)*F] = temp8;
			xx[index + tile_x + (tile_y + 9)*F] = temp9;

			xx[index + tile_x + 1 + tile_y*F] = temp10;
			xx[index + tile_x + 1 + (tile_y + 1)*F] = temp11;
			xx[index + tile_x + 1 + (tile_y + 2)*F] = temp12;
			xx[index + tile_x + 1 + (tile_y + 3)*F] = temp13;
			xx[index + tile_x + 1 + (tile_y + 4)*F] = temp14;
			xx[index + tile_x + 1 + (tile_y + 5)*F] = temp15;
			xx[index + tile_x + 1 + (tile_y + 6)*F] = temp16;
			xx[index + tile_x + 1 + (tile_y + 7)*F] = temp17;
			xx[index + tile_x + 1 + (tile_y + 8)*F] = temp18;
			xx[index + tile_x + 1 + (tile_y + 9)*F] = temp19;

			xx[index + tile_x + 2 + tile_y*F] = temp20;
			xx[index + tile_x + 2 + (tile_y + 1)*F] = temp21;
			xx[index + tile_x + 2 + (tile_y + 2)*F] = temp22;
			xx[index + tile_x + 2 + (tile_y + 3)*F] = temp23;
			xx[index + tile_x + 2 + (tile_y + 4)*F] = temp24;
			xx[index + tile_x + 2 + (tile_y + 5)*F] = temp25;
			xx[index + tile_x + 2 + (tile_y + 6)*F] = temp26;
			xx[index + tile_x + 2 + (tile_y + 7)*F] = temp27;
			xx[index + tile_x + 2 + (tile_y + 8)*F] = temp28;
			xx[index + tile_x + 2 + (tile_y + 9)*F] = temp29;

			xx[index + tile_x + 3 + tile_y*F] = temp30;
			xx[index + tile_x + 3 + (tile_y + 1)*F] = temp31;
			xx[index + tile_x + 3 + (tile_y + 2)*F] = temp32;
			xx[index + tile_x + 3 + (tile_y + 3)*F] = temp33;
			xx[index + tile_x + 3 + (tile_y + 4)*F] = temp34;
			xx[index + tile_x + 3 + (tile_y + 5)*F] = temp35;
			xx[index + tile_x + 3 + (tile_y + 6)*F] = temp36;
			xx[index + tile_x + 3 + (tile_y + 7)*F] = temp37;
			xx[index + tile_x + 3 + (tile_y + 8)*F] = temp38;
			xx[index + tile_x + 3 + (tile_y + 9)*F] = temp39;

			xx[index + tile_x + 4 + tile_y*F] = temp40;
			xx[index + tile_x + 4 + (tile_y + 1)*F] = temp41;
			xx[index + tile_x + 4 + (tile_y + 2)*F] = temp42;
			xx[index + tile_x + 4 + (tile_y + 3)*F] = temp43;
			xx[index + tile_x + 4 + (tile_y + 4)*F] = temp44;
			xx[index + tile_x + 4 + (tile_y + 5)*F] = temp45;
			xx[index + tile_x + 4 + (tile_y + 6)*F] = temp46;
			xx[index + tile_x + 4 + (tile_y + 7)*F] = temp47;
			xx[index + tile_x + 4 + (tile_y + 8)*F] = temp48;
			xx[index + tile_x + 4 + (tile_y + 9)*F] = temp49;

			xx[index + tile_x + 5 + tile_y*F] = temp50;
			xx[index + tile_x + 5 + (tile_y + 1)*F] = temp51;
			xx[index + tile_x + 5 + (tile_y + 2)*F] = temp52;
			xx[index + tile_x + 5 + (tile_y + 3)*F] = temp53;
			xx[index + tile_x + 5 + (tile_y + 4)*F] = temp54;
			xx[index + tile_x + 5 + (tile_y + 5)*F] = temp55;
			xx[index + tile_x + 5 + (tile_y + 6)*F] = temp56;
			xx[index + tile_x + 5 + (tile_y + 7)*F] = temp57;
			xx[index + tile_x + 5 + (tile_y + 8)*F] = temp58;
			xx[index + tile_x + 5 + (tile_y + 9)*F] = temp59;

			xx[index + tile_x + 6 + tile_y*F] = temp60;
			xx[index + tile_x + 6 + (tile_y + 1)*F] = temp61;
			xx[index + tile_x + 6 + (tile_y + 2)*F] = temp62;
			xx[index + tile_x + 6 + (tile_y + 3)*F] = temp63;
			xx[index + tile_x + 6 + (tile_y + 4)*F] = temp64;
			xx[index + tile_x + 6 + (tile_y + 5)*F] = temp65;
			xx[index + tile_x + 6 + (tile_y + 6)*F] = temp66;
			xx[index + tile_x + 6 + (tile_y + 7)*F] = temp67;
			xx[index + tile_x + 6 + (tile_y + 8)*F] = temp68;
			xx[index + tile_x + 6 + (tile_y + 9)*F] = temp69;

			xx[index + tile_x + 7 + tile_y*F] = temp70;
			xx[index + tile_x + 7 + (tile_y + 1)*F] = temp71;
			xx[index + tile_x + 7 + (tile_y + 2)*F] = temp72;
			xx[index + tile_x + 7 + (tile_y + 3)*F] = temp73;
			xx[index + tile_x + 7 + (tile_y + 4)*F] = temp74;
			xx[index + tile_x + 7 + (tile_y + 5)*F] = temp75;
			xx[index + tile_x + 7 + (tile_y + 6)*F] = temp76;
			xx[index + tile_x + 7 + (tile_y + 7)*F] = temp77;
			xx[index + tile_x + 7 + (tile_y + 8)*F] = temp78;
			xx[index + tile_x + 7 + (tile_y + 9)*F] = temp79;

			xx[index + tile_x + 8 + tile_y*F] = temp80;
			xx[index + tile_x + 8 + (tile_y + 1)*F] = temp81;
			xx[index + tile_x + 8 + (tile_y + 2)*F] = temp82;
			xx[index + tile_x + 8 + (tile_y + 3)*F] = temp83;
			xx[index + tile_x + 8 + (tile_y + 4)*F] = temp84;
			xx[index + tile_x + 8 + (tile_y + 5)*F] = temp85;
			xx[index + tile_x + 8 + (tile_y + 6)*F] = temp86;
			xx[index + tile_x + 8 + (tile_y + 7)*F] = temp87;
			xx[index + tile_x + 8 + (tile_y + 8)*F] = temp88;
			xx[index + tile_x + 8 + (tile_y + 9)*F] = temp89;

			xx[index + tile_x + 9 + tile_y*F] = temp90;
			xx[index + tile_x + 9 + (tile_y + 1)*F] = temp91;
			xx[index + tile_x + 9 + (tile_y + 2)*F] = temp92;
			xx[index + tile_x + 9 + (tile_y + 3)*F] = temp93;
			xx[index + tile_x + 9 + (tile_y + 4)*F] = temp94;
			xx[index + tile_x + 9 + (tile_y + 5)*F] = temp95;
			xx[index + tile_x + 9 + (tile_y + 6)*F] = temp96;
			xx[index + tile_x + 9 + (tile_y + 7)*F] = temp97;
			xx[index + tile_x + 9 + (tile_y + 8)*F] = temp98;
			xx[index + tile_x + 9 + (tile_y + 9)*F] = temp99;
	//*/
			//regularization
			if(tile_x == tile_y){
				for(int k = 0; k < tile; k++)
					xx[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}
}



__global__ void
__launch_bounds__(64, 6)
get_hermitian_x(float* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda) {
	__shared__ float2 thetaTemp[SCAN_BATCH * F/2];
	int row = blockIdx.x;
	if (row < M) {
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


		//int tile_x = (threadIdx.x/tile) * tile;//start x of this tile
		//int tile_y = (threadIdx.x%tile) * tile;//start y of this tile
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
			//32 > SCAN_BATCH
			if(threadIdx.x < 2*32 ){
				//int index = threadIdx.x;
				int index = threadIdx.x - (threadIdx.x/32)*32;	//0 to 31;
				if(index < SCAN_BATCH){
					if(iter*SCAN_BATCH + index < end - start){
						//for (int k = 50*(threadIdx.x/32); k < 50*(threadIdx.x/32) + 50; k += 2){
						//IMPORTANT: for loop has constant and identical start and end
						if(threadIdx.x < 32){
							for (int k = 0; k < 50; k += 2){
								theta.x = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + index] + k);
								theta.y = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + index] + k+1);
								thetaTemp[index * F/2 + k/2] = theta;
							}
						}
						else {
							for (int k = 0; k < 50; k += 2){
								theta.x = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 50);
								theta.y = tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 51);
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
			///////////////////////////////////////////////////////////////////////////////////////////////////////////

			//tile: 10*10
			if(threadIdx.x < 55 ){
				for(int k = 0; k < SCAN_BATCH; k++){
					temp0 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp1 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp2 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp3 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp4 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp5 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp6 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp7 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp8 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp9 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp10 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp11 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp12 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp13 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp14 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp15 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp16 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp17 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp18 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp19 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp20 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp21 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp22 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp23 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp24 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp25 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp26 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp27 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp28 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp29 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp30 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp31 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp32 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp33 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp34 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp35 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp36 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp37 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp38 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp39 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp40 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp41 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp42 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp43 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp44 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp45 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp46 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp47 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp48 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp49 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp50 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp51 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp52 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp53 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp54 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp55 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp56 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp57 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp58 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp59 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp60 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp61 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp62 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp63 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp64 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp65 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp66 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp67 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp68 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp69 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp70 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp71 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp72 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp73 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp74 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp75 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp76 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp77 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp78 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp79 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;


					temp80 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp81 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp82 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp83 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp84 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp85 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp86 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp87 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp88 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp89 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp90 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp91 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp92 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp93 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp94 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp95 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp96 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp97 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp98 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp99 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
			///////////////////////////////////////////////////////////////////////////////////////////////////////////
		__syncthreads();
///*
		if(threadIdx.x < 55 ){
			//copy output to gmem
			int index = blockIdx.x*F*F;
			tt[index + tile_x + tile_y*F] = temp0;
			tt[index + tile_x + (tile_y + 1)*F] = temp1;
			tt[index + tile_x + (tile_y + 2)*F] = temp2;
			tt[index + tile_x + (tile_y + 3)*F] = temp3;
			tt[index + tile_x + (tile_y + 4)*F] = temp4;
			tt[index + tile_x + (tile_y + 5)*F] = temp5;
			tt[index + tile_x + (tile_y + 6)*F] = temp6;
			tt[index + tile_x + (tile_y + 7)*F] = temp7;
			tt[index + tile_x + (tile_y + 8)*F] = temp8;
			tt[index + tile_x + (tile_y + 9)*F] = temp9;

			tt[index + tile_x + 1 + tile_y*F] = temp10;
			tt[index + tile_x + 1 + (tile_y + 1)*F] = temp11;
			tt[index + tile_x + 1 + (tile_y + 2)*F] = temp12;
			tt[index + tile_x + 1 + (tile_y + 3)*F] = temp13;
			tt[index + tile_x + 1 + (tile_y + 4)*F] = temp14;
			tt[index + tile_x + 1 + (tile_y + 5)*F] = temp15;
			tt[index + tile_x + 1 + (tile_y + 6)*F] = temp16;
			tt[index + tile_x + 1 + (tile_y + 7)*F] = temp17;
			tt[index + tile_x + 1 + (tile_y + 8)*F] = temp18;
			tt[index + tile_x + 1 + (tile_y + 9)*F] = temp19;

			tt[index + tile_x + 2 + tile_y*F] = temp20;
			tt[index + tile_x + 2 + (tile_y + 1)*F] = temp21;
			tt[index + tile_x + 2 + (tile_y + 2)*F] = temp22;
			tt[index + tile_x + 2 + (tile_y + 3)*F] = temp23;
			tt[index + tile_x + 2 + (tile_y + 4)*F] = temp24;
			tt[index + tile_x + 2 + (tile_y + 5)*F] = temp25;
			tt[index + tile_x + 2 + (tile_y + 6)*F] = temp26;
			tt[index + tile_x + 2 + (tile_y + 7)*F] = temp27;
			tt[index + tile_x + 2 + (tile_y + 8)*F] = temp28;
			tt[index + tile_x + 2 + (tile_y + 9)*F] = temp29;

			tt[index + tile_x + 3 + tile_y*F] = temp30;
			tt[index + tile_x + 3 + (tile_y + 1)*F] = temp31;
			tt[index + tile_x + 3 + (tile_y + 2)*F] = temp32;
			tt[index + tile_x + 3 + (tile_y + 3)*F] = temp33;
			tt[index + tile_x + 3 + (tile_y + 4)*F] = temp34;
			tt[index + tile_x + 3 + (tile_y + 5)*F] = temp35;
			tt[index + tile_x + 3 + (tile_y + 6)*F] = temp36;
			tt[index + tile_x + 3 + (tile_y + 7)*F] = temp37;
			tt[index + tile_x + 3 + (tile_y + 8)*F] = temp38;
			tt[index + tile_x + 3 + (tile_y + 9)*F] = temp39;

			tt[index + tile_x + 4 + tile_y*F] = temp40;
			tt[index + tile_x + 4 + (tile_y + 1)*F] = temp41;
			tt[index + tile_x + 4 + (tile_y + 2)*F] = temp42;
			tt[index + tile_x + 4 + (tile_y + 3)*F] = temp43;
			tt[index + tile_x + 4 + (tile_y + 4)*F] = temp44;
			tt[index + tile_x + 4 + (tile_y + 5)*F] = temp45;
			tt[index + tile_x + 4 + (tile_y + 6)*F] = temp46;
			tt[index + tile_x + 4 + (tile_y + 7)*F] = temp47;
			tt[index + tile_x + 4 + (tile_y + 8)*F] = temp48;
			tt[index + tile_x + 4 + (tile_y + 9)*F] = temp49;

			tt[index + tile_x + 5 + tile_y*F] = temp50;
			tt[index + tile_x + 5 + (tile_y + 1)*F] = temp51;
			tt[index + tile_x + 5 + (tile_y + 2)*F] = temp52;
			tt[index + tile_x + 5 + (tile_y + 3)*F] = temp53;
			tt[index + tile_x + 5 + (tile_y + 4)*F] = temp54;
			tt[index + tile_x + 5 + (tile_y + 5)*F] = temp55;
			tt[index + tile_x + 5 + (tile_y + 6)*F] = temp56;
			tt[index + tile_x + 5 + (tile_y + 7)*F] = temp57;
			tt[index + tile_x + 5 + (tile_y + 8)*F] = temp58;
			tt[index + tile_x + 5 + (tile_y + 9)*F] = temp59;

			tt[index + tile_x + 6 + tile_y*F] = temp60;
			tt[index + tile_x + 6 + (tile_y + 1)*F] = temp61;
			tt[index + tile_x + 6 + (tile_y + 2)*F] = temp62;
			tt[index + tile_x + 6 + (tile_y + 3)*F] = temp63;
			tt[index + tile_x + 6 + (tile_y + 4)*F] = temp64;
			tt[index + tile_x + 6 + (tile_y + 5)*F] = temp65;
			tt[index + tile_x + 6 + (tile_y + 6)*F] = temp66;
			tt[index + tile_x + 6 + (tile_y + 7)*F] = temp67;
			tt[index + tile_x + 6 + (tile_y + 8)*F] = temp68;
			tt[index + tile_x + 6 + (tile_y + 9)*F] = temp69;

			tt[index + tile_x + 7 + tile_y*F] = temp70;
			tt[index + tile_x + 7 + (tile_y + 1)*F] = temp71;
			tt[index + tile_x + 7 + (tile_y + 2)*F] = temp72;
			tt[index + tile_x + 7 + (tile_y + 3)*F] = temp73;
			tt[index + tile_x + 7 + (tile_y + 4)*F] = temp74;
			tt[index + tile_x + 7 + (tile_y + 5)*F] = temp75;
			tt[index + tile_x + 7 + (tile_y + 6)*F] = temp76;
			tt[index + tile_x + 7 + (tile_y + 7)*F] = temp77;
			tt[index + tile_x + 7 + (tile_y + 8)*F] = temp78;
			tt[index + tile_x + 7 + (tile_y + 9)*F] = temp79;

			tt[index + tile_x + 8 + tile_y*F] = temp80;
			tt[index + tile_x + 8 + (tile_y + 1)*F] = temp81;
			tt[index + tile_x + 8 + (tile_y + 2)*F] = temp82;
			tt[index + tile_x + 8 + (tile_y + 3)*F] = temp83;
			tt[index + tile_x + 8 + (tile_y + 4)*F] = temp84;
			tt[index + tile_x + 8 + (tile_y + 5)*F] = temp85;
			tt[index + tile_x + 8 + (tile_y + 6)*F] = temp86;
			tt[index + tile_x + 8 + (tile_y + 7)*F] = temp87;
			tt[index + tile_x + 8 + (tile_y + 8)*F] = temp88;
			tt[index + tile_x + 8 + (tile_y + 9)*F] = temp89;

			tt[index + tile_x + 9 + tile_y*F] = temp90;
			tt[index + tile_x + 9 + (tile_y + 1)*F] = temp91;
			tt[index + tile_x + 9 + (tile_y + 2)*F] = temp92;
			tt[index + tile_x + 9 + (tile_y + 3)*F] = temp93;
			tt[index + tile_x + 9 + (tile_y + 4)*F] = temp94;
			tt[index + tile_x + 9 + (tile_y + 5)*F] = temp95;
			tt[index + tile_x + 9 + (tile_y + 6)*F] = temp96;
			tt[index + tile_x + 9 + (tile_y + 7)*F] = temp97;
			tt[index + tile_x + 9 + (tile_y + 8)*F] = temp98;
			tt[index + tile_x + 9 + (tile_y + 9)*F] = temp99;

			//symmetric
			if(tile_x!=tile_y){
				tt[index + tile_y + 0+ (tile_x + 0)*F]= temp0;
				tt[index + tile_y + 1+ (tile_x + 0)*F]= temp1;
				tt[index + tile_y + 2+ (tile_x + 0)*F]= temp2;
				tt[index + tile_y + 3+ (tile_x + 0)*F]= temp3;
				tt[index + tile_y + 4+ (tile_x + 0)*F]= temp4;
				tt[index + tile_y + 5+ (tile_x + 0)*F]= temp5;
				tt[index + tile_y + 6+ (tile_x + 0)*F]= temp6;
				tt[index + tile_y + 7+ (tile_x + 0)*F]= temp7;
				tt[index + tile_y + 8+ (tile_x + 0)*F]= temp8;
				tt[index + tile_y + 9+ (tile_x + 0)*F]= temp9;


				tt[index + tile_y + 0+ (tile_x + 1)*F]= temp10;
				tt[index + tile_y + 1+ (tile_x + 1)*F]= temp11;
				tt[index + tile_y + 2+ (tile_x + 1)*F]= temp12;
				tt[index + tile_y + 3+ (tile_x + 1)*F]= temp13;
				tt[index + tile_y + 4+ (tile_x + 1)*F]= temp14;
				tt[index + tile_y + 5+ (tile_x + 1)*F]= temp15;
				tt[index + tile_y + 6+ (tile_x + 1)*F]= temp16;
				tt[index + tile_y + 7+ (tile_x + 1)*F]= temp17;
				tt[index + tile_y + 8+ (tile_x + 1)*F]= temp18;
				tt[index + tile_y + 9+ (tile_x + 1)*F]= temp19;


				tt[index + tile_y + 0+ (tile_x + 2)*F]= temp20;
				tt[index + tile_y + 1+ (tile_x + 2)*F]= temp21;
				tt[index + tile_y + 2+ (tile_x + 2)*F]= temp22;
				tt[index + tile_y + 3+ (tile_x + 2)*F]= temp23;
				tt[index + tile_y + 4+ (tile_x + 2)*F]= temp24;
				tt[index + tile_y + 5+ (tile_x + 2)*F]= temp25;
				tt[index + tile_y + 6+ (tile_x + 2)*F]= temp26;
				tt[index + tile_y + 7+ (tile_x + 2)*F]= temp27;
				tt[index + tile_y + 8+ (tile_x + 2)*F]= temp28;
				tt[index + tile_y + 9+ (tile_x + 2)*F]= temp29;


				tt[index + tile_y + 0+ (tile_x + 3)*F]= temp30;
				tt[index + tile_y + 1+ (tile_x + 3)*F]= temp31;
				tt[index + tile_y + 2+ (tile_x + 3)*F]= temp32;
				tt[index + tile_y + 3+ (tile_x + 3)*F]= temp33;
				tt[index + tile_y + 4+ (tile_x + 3)*F]= temp34;
				tt[index + tile_y + 5+ (tile_x + 3)*F]= temp35;
				tt[index + tile_y + 6+ (tile_x + 3)*F]= temp36;
				tt[index + tile_y + 7+ (tile_x + 3)*F]= temp37;
				tt[index + tile_y + 8+ (tile_x + 3)*F]= temp38;
				tt[index + tile_y + 9+ (tile_x + 3)*F]= temp39;


				tt[index + tile_y + 0+ (tile_x + 4)*F]= temp40;
				tt[index + tile_y + 1+ (tile_x + 4)*F]= temp41;
				tt[index + tile_y + 2+ (tile_x + 4)*F]= temp42;
				tt[index + tile_y + 3+ (tile_x + 4)*F]= temp43;
				tt[index + tile_y + 4+ (tile_x + 4)*F]= temp44;
				tt[index + tile_y + 5+ (tile_x + 4)*F]= temp45;
				tt[index + tile_y + 6+ (tile_x + 4)*F]= temp46;
				tt[index + tile_y + 7+ (tile_x + 4)*F]= temp47;
				tt[index + tile_y + 8+ (tile_x + 4)*F]= temp48;
				tt[index + tile_y + 9+ (tile_x + 4)*F]= temp49;


				tt[index + tile_y + 0+ (tile_x + 5)*F]= temp50;
				tt[index + tile_y + 1+ (tile_x + 5)*F]= temp51;
				tt[index + tile_y + 2+ (tile_x + 5)*F]= temp52;
				tt[index + tile_y + 3+ (tile_x + 5)*F]= temp53;
				tt[index + tile_y + 4+ (tile_x + 5)*F]= temp54;
				tt[index + tile_y + 5+ (tile_x + 5)*F]= temp55;
				tt[index + tile_y + 6+ (tile_x + 5)*F]= temp56;
				tt[index + tile_y + 7+ (tile_x + 5)*F]= temp57;
				tt[index + tile_y + 8+ (tile_x + 5)*F]= temp58;
				tt[index + tile_y + 9+ (tile_x + 5)*F]= temp59;


				tt[index + tile_y + 0+ (tile_x + 6)*F]= temp60;
				tt[index + tile_y + 1+ (tile_x + 6)*F]= temp61;
				tt[index + tile_y + 2+ (tile_x + 6)*F]= temp62;
				tt[index + tile_y + 3+ (tile_x + 6)*F]= temp63;
				tt[index + tile_y + 4+ (tile_x + 6)*F]= temp64;
				tt[index + tile_y + 5+ (tile_x + 6)*F]= temp65;
				tt[index + tile_y + 6+ (tile_x + 6)*F]= temp66;
				tt[index + tile_y + 7+ (tile_x + 6)*F]= temp67;
				tt[index + tile_y + 8+ (tile_x + 6)*F]= temp68;
				tt[index + tile_y + 9+ (tile_x + 6)*F]= temp69;


				tt[index + tile_y + 0+ (tile_x + 7)*F]= temp70;
				tt[index + tile_y + 1+ (tile_x + 7)*F]= temp71;
				tt[index + tile_y + 2+ (tile_x + 7)*F]= temp72;
				tt[index + tile_y + 3+ (tile_x + 7)*F]= temp73;
				tt[index + tile_y + 4+ (tile_x + 7)*F]= temp74;
				tt[index + tile_y + 5+ (tile_x + 7)*F]= temp75;
				tt[index + tile_y + 6+ (tile_x + 7)*F]= temp76;
				tt[index + tile_y + 7+ (tile_x + 7)*F]= temp77;
				tt[index + tile_y + 8+ (tile_x + 7)*F]= temp78;
				tt[index + tile_y + 9+ (tile_x + 7)*F]= temp79;


				tt[index + tile_y + 0+ (tile_x + 8)*F]= temp80;
				tt[index + tile_y + 1+ (tile_x + 8)*F]= temp81;
				tt[index + tile_y + 2+ (tile_x + 8)*F]= temp82;
				tt[index + tile_y + 3+ (tile_x + 8)*F]= temp83;
				tt[index + tile_y + 4+ (tile_x + 8)*F]= temp84;
				tt[index + tile_y + 5+ (tile_x + 8)*F]= temp85;
				tt[index + tile_y + 6+ (tile_x + 8)*F]= temp86;
				tt[index + tile_y + 7+ (tile_x + 8)*F]= temp87;
				tt[index + tile_y + 8+ (tile_x + 8)*F]= temp88;
				tt[index + tile_y + 9+ (tile_x + 8)*F]= temp89;


				tt[index + tile_y + 0+ (tile_x + 9)*F]= temp90;
				tt[index + tile_y + 1+ (tile_x + 9)*F]= temp91;
				tt[index + tile_y + 2+ (tile_x + 9)*F]= temp92;
				tt[index + tile_y + 3+ (tile_x + 9)*F]= temp93;
				tt[index + tile_y + 4+ (tile_x + 9)*F]= temp94;
				tt[index + tile_y + 5+ (tile_x + 9)*F]= temp95;
				tt[index + tile_y + 6+ (tile_x + 9)*F]= temp96;
				tt[index + tile_y + 7+ (tile_x + 9)*F]= temp97;
				tt[index + tile_y + 8+ (tile_x + 9)*F]= temp98;
				tt[index + tile_y + 9+ (tile_x + 9)*F]= temp99;
			}
			//regularization
			if(tile_x == tile_y){
				for(int k = 0; k < tile; k++)
					tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}


//*/

	}
}


__global__ void
__launch_bounds__(64, 6)
get_hermitian_theta(float* xx,
		const int* cscRowIndex, const int* cscColIndex, const float lambda, const float * XT) {
	__shared__ float2 xTemp[SCAN_BATCH * F/2];
	int col = blockIdx.x;
	if (col < N) {
		//this block needs to handle end - start thetaT columns
		int start = cscColIndex[col];
		int end = cscColIndex[col + 1];
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


		//int tile_x = (threadIdx.x/tile) * tile;//start x of this tile
		//int tile_y = (threadIdx.x%tile) * tile;//start y of this tile
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
			float2 x;
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
			//32 > SCAN_BATCH
			if(threadIdx.x < 2*32 ){
				//int index = threadIdx.x;
				int index = threadIdx.x - (threadIdx.x/32)*32;	//0 to 31;
				if(index < SCAN_BATCH){
					if(iter*SCAN_BATCH + index < end - start){
						//for (int k = 50*(threadIdx.x/32); k < 50*(threadIdx.x/32) + 50; k += 2){
						//IMPORTANT: for loop has constant and identical start and end
						if(threadIdx.x < 32){
							for (int k = 0; k < 50; k += 2){
								x.x = XT[ F * cscRowIndex[start + iter*SCAN_BATCH + index] + k ];
								x.y = XT[ F * cscRowIndex[start + iter*SCAN_BATCH + index] + k+1];
								xTemp[index * F/2 + k/2] = x;
							}
						}
						else {
							for (int k = 0; k < 50; k += 2){
								x.x = XT[ F * cscRowIndex[start + iter*SCAN_BATCH + index] + k + 50];
								x.y = XT[ F * cscRowIndex[start + iter*SCAN_BATCH + index] + k + 51];
								xTemp[index * F/2 + k/2 + 25] = x;
							}
						}
					}
					//must be the last iteration; no need to check
					//not enough theta to copy, set zero
					else
						memset(&xTemp[index*F/2], 0, F*sizeof(float));
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
			///////////////////////////////////////////////////////////////////////////////////////////////////////////

			//tile: 10*10
			if(threadIdx.x < 55 ){
				for(int k = 0; k < SCAN_BATCH; k++){
					temp0 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp1 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp2 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp3 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp4 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp5 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp6 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp7 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp8 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp9 += xTemp[tile_x/2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp10 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp11 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp12 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp13 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp14 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp15 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp16 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp17 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp18 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp19 += xTemp[tile_x/2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;

					temp20 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp21 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp22 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp23 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp24 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp25 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp26 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp27 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp28 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp29 += xTemp[tile_x/2 +1 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp30 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp31 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp32 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp33 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp34 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp35 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp36 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp37 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp38 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp39 += xTemp[tile_x/2 +1 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;

					temp40 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp41 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp42 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp43 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp44 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp45 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp46 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp47 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp48 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp49 += xTemp[tile_x/2 +2 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp50 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp51 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp52 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp53 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp54 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp55 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp56 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp57 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp58 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp59 += xTemp[tile_x/2 +2 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;

					temp60 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp61 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp62 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp63 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp64 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp65 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp66 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp67 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp68 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp69 += xTemp[tile_x/2 +3 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp70 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp71 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp72 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp73 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp74 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp75 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp76 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp77 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp78 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp79 += xTemp[tile_x/2 +3 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;


					temp80 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 + k*F/2].x;
					temp81 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 + k*F/2].y;
					temp82 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].x;
					temp83 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +1 + k*F/2].y;
					temp84 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].x;
					temp85 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +2 + k*F/2].y;
					temp86 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].x;
					temp87 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +3 + k*F/2].y;
					temp88 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].x;
					temp89 += xTemp[tile_x/2 +4 + k*F/2].x * xTemp[tile_y/2 +4 + k*F/2].y;

					temp90 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 + k*F/2].x;
					temp91 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 + k*F/2].y;
					temp92 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].x;
					temp93 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +1 + k*F/2].y;
					temp94 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].x;
					temp95 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +2 + k*F/2].y;
					temp96 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].x;
					temp97 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +3 + k*F/2].y;
					temp98 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].x;
					temp99 += xTemp[tile_x/2 +4 + k*F/2].y * xTemp[tile_y/2 +4 + k*F/2].y;
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
			///////////////////////////////////////////////////////////////////////////////////////////////////////////
		__syncthreads();
///*
		if(threadIdx.x < 55 ){
			//copy output to gmem
			int index = blockIdx.x*F*F;
			xx[index + tile_x + tile_y*F] = temp0;
			xx[index + tile_x + (tile_y + 1)*F] = temp1;
			xx[index + tile_x + (tile_y + 2)*F] = temp2;
			xx[index + tile_x + (tile_y + 3)*F] = temp3;
			xx[index + tile_x + (tile_y + 4)*F] = temp4;
			xx[index + tile_x + (tile_y + 5)*F] = temp5;
			xx[index + tile_x + (tile_y + 6)*F] = temp6;
			xx[index + tile_x + (tile_y + 7)*F] = temp7;
			xx[index + tile_x + (tile_y + 8)*F] = temp8;
			xx[index + tile_x + (tile_y + 9)*F] = temp9;

			xx[index + tile_x + 1 + tile_y*F] = temp10;
			xx[index + tile_x + 1 + (tile_y + 1)*F] = temp11;
			xx[index + tile_x + 1 + (tile_y + 2)*F] = temp12;
			xx[index + tile_x + 1 + (tile_y + 3)*F] = temp13;
			xx[index + tile_x + 1 + (tile_y + 4)*F] = temp14;
			xx[index + tile_x + 1 + (tile_y + 5)*F] = temp15;
			xx[index + tile_x + 1 + (tile_y + 6)*F] = temp16;
			xx[index + tile_x + 1 + (tile_y + 7)*F] = temp17;
			xx[index + tile_x + 1 + (tile_y + 8)*F] = temp18;
			xx[index + tile_x + 1 + (tile_y + 9)*F] = temp19;

			xx[index + tile_x + 2 + tile_y*F] = temp20;
			xx[index + tile_x + 2 + (tile_y + 1)*F] = temp21;
			xx[index + tile_x + 2 + (tile_y + 2)*F] = temp22;
			xx[index + tile_x + 2 + (tile_y + 3)*F] = temp23;
			xx[index + tile_x + 2 + (tile_y + 4)*F] = temp24;
			xx[index + tile_x + 2 + (tile_y + 5)*F] = temp25;
			xx[index + tile_x + 2 + (tile_y + 6)*F] = temp26;
			xx[index + tile_x + 2 + (tile_y + 7)*F] = temp27;
			xx[index + tile_x + 2 + (tile_y + 8)*F] = temp28;
			xx[index + tile_x + 2 + (tile_y + 9)*F] = temp29;

			xx[index + tile_x + 3 + tile_y*F] = temp30;
			xx[index + tile_x + 3 + (tile_y + 1)*F] = temp31;
			xx[index + tile_x + 3 + (tile_y + 2)*F] = temp32;
			xx[index + tile_x + 3 + (tile_y + 3)*F] = temp33;
			xx[index + tile_x + 3 + (tile_y + 4)*F] = temp34;
			xx[index + tile_x + 3 + (tile_y + 5)*F] = temp35;
			xx[index + tile_x + 3 + (tile_y + 6)*F] = temp36;
			xx[index + tile_x + 3 + (tile_y + 7)*F] = temp37;
			xx[index + tile_x + 3 + (tile_y + 8)*F] = temp38;
			xx[index + tile_x + 3 + (tile_y + 9)*F] = temp39;

			xx[index + tile_x + 4 + tile_y*F] = temp40;
			xx[index + tile_x + 4 + (tile_y + 1)*F] = temp41;
			xx[index + tile_x + 4 + (tile_y + 2)*F] = temp42;
			xx[index + tile_x + 4 + (tile_y + 3)*F] = temp43;
			xx[index + tile_x + 4 + (tile_y + 4)*F] = temp44;
			xx[index + tile_x + 4 + (tile_y + 5)*F] = temp45;
			xx[index + tile_x + 4 + (tile_y + 6)*F] = temp46;
			xx[index + tile_x + 4 + (tile_y + 7)*F] = temp47;
			xx[index + tile_x + 4 + (tile_y + 8)*F] = temp48;
			xx[index + tile_x + 4 + (tile_y + 9)*F] = temp49;

			xx[index + tile_x + 5 + tile_y*F] = temp50;
			xx[index + tile_x + 5 + (tile_y + 1)*F] = temp51;
			xx[index + tile_x + 5 + (tile_y + 2)*F] = temp52;
			xx[index + tile_x + 5 + (tile_y + 3)*F] = temp53;
			xx[index + tile_x + 5 + (tile_y + 4)*F] = temp54;
			xx[index + tile_x + 5 + (tile_y + 5)*F] = temp55;
			xx[index + tile_x + 5 + (tile_y + 6)*F] = temp56;
			xx[index + tile_x + 5 + (tile_y + 7)*F] = temp57;
			xx[index + tile_x + 5 + (tile_y + 8)*F] = temp58;
			xx[index + tile_x + 5 + (tile_y + 9)*F] = temp59;

			xx[index + tile_x + 6 + tile_y*F] = temp60;
			xx[index + tile_x + 6 + (tile_y + 1)*F] = temp61;
			xx[index + tile_x + 6 + (tile_y + 2)*F] = temp62;
			xx[index + tile_x + 6 + (tile_y + 3)*F] = temp63;
			xx[index + tile_x + 6 + (tile_y + 4)*F] = temp64;
			xx[index + tile_x + 6 + (tile_y + 5)*F] = temp65;
			xx[index + tile_x + 6 + (tile_y + 6)*F] = temp66;
			xx[index + tile_x + 6 + (tile_y + 7)*F] = temp67;
			xx[index + tile_x + 6 + (tile_y + 8)*F] = temp68;
			xx[index + tile_x + 6 + (tile_y + 9)*F] = temp69;

			xx[index + tile_x + 7 + tile_y*F] = temp70;
			xx[index + tile_x + 7 + (tile_y + 1)*F] = temp71;
			xx[index + tile_x + 7 + (tile_y + 2)*F] = temp72;
			xx[index + tile_x + 7 + (tile_y + 3)*F] = temp73;
			xx[index + tile_x + 7 + (tile_y + 4)*F] = temp74;
			xx[index + tile_x + 7 + (tile_y + 5)*F] = temp75;
			xx[index + tile_x + 7 + (tile_y + 6)*F] = temp76;
			xx[index + tile_x + 7 + (tile_y + 7)*F] = temp77;
			xx[index + tile_x + 7 + (tile_y + 8)*F] = temp78;
			xx[index + tile_x + 7 + (tile_y + 9)*F] = temp79;

			xx[index + tile_x + 8 + tile_y*F] = temp80;
			xx[index + tile_x + 8 + (tile_y + 1)*F] = temp81;
			xx[index + tile_x + 8 + (tile_y + 2)*F] = temp82;
			xx[index + tile_x + 8 + (tile_y + 3)*F] = temp83;
			xx[index + tile_x + 8 + (tile_y + 4)*F] = temp84;
			xx[index + tile_x + 8 + (tile_y + 5)*F] = temp85;
			xx[index + tile_x + 8 + (tile_y + 6)*F] = temp86;
			xx[index + tile_x + 8 + (tile_y + 7)*F] = temp87;
			xx[index + tile_x + 8 + (tile_y + 8)*F] = temp88;
			xx[index + tile_x + 8 + (tile_y + 9)*F] = temp89;

			xx[index + tile_x + 9 + tile_y*F] = temp90;
			xx[index + tile_x + 9 + (tile_y + 1)*F] = temp91;
			xx[index + tile_x + 9 + (tile_y + 2)*F] = temp92;
			xx[index + tile_x + 9 + (tile_y + 3)*F] = temp93;
			xx[index + tile_x + 9 + (tile_y + 4)*F] = temp94;
			xx[index + tile_x + 9 + (tile_y + 5)*F] = temp95;
			xx[index + tile_x + 9 + (tile_y + 6)*F] = temp96;
			xx[index + tile_x + 9 + (tile_y + 7)*F] = temp97;
			xx[index + tile_x + 9 + (tile_y + 8)*F] = temp98;
			xx[index + tile_x + 9 + (tile_y + 9)*F] = temp99;

			//symmetric
			if(tile_x!=tile_y){
				xx[index + tile_y + 0+ (tile_x + 0)*F]= temp0;
				xx[index + tile_y + 1+ (tile_x + 0)*F]= temp1;
				xx[index + tile_y + 2+ (tile_x + 0)*F]= temp2;
				xx[index + tile_y + 3+ (tile_x + 0)*F]= temp3;
				xx[index + tile_y + 4+ (tile_x + 0)*F]= temp4;
				xx[index + tile_y + 5+ (tile_x + 0)*F]= temp5;
				xx[index + tile_y + 6+ (tile_x + 0)*F]= temp6;
				xx[index + tile_y + 7+ (tile_x + 0)*F]= temp7;
				xx[index + tile_y + 8+ (tile_x + 0)*F]= temp8;
				xx[index + tile_y + 9+ (tile_x + 0)*F]= temp9;


				xx[index + tile_y + 0+ (tile_x + 1)*F]= temp10;
				xx[index + tile_y + 1+ (tile_x + 1)*F]= temp11;
				xx[index + tile_y + 2+ (tile_x + 1)*F]= temp12;
				xx[index + tile_y + 3+ (tile_x + 1)*F]= temp13;
				xx[index + tile_y + 4+ (tile_x + 1)*F]= temp14;
				xx[index + tile_y + 5+ (tile_x + 1)*F]= temp15;
				xx[index + tile_y + 6+ (tile_x + 1)*F]= temp16;
				xx[index + tile_y + 7+ (tile_x + 1)*F]= temp17;
				xx[index + tile_y + 8+ (tile_x + 1)*F]= temp18;
				xx[index + tile_y + 9+ (tile_x + 1)*F]= temp19;


				xx[index + tile_y + 0+ (tile_x + 2)*F]= temp20;
				xx[index + tile_y + 1+ (tile_x + 2)*F]= temp21;
				xx[index + tile_y + 2+ (tile_x + 2)*F]= temp22;
				xx[index + tile_y + 3+ (tile_x + 2)*F]= temp23;
				xx[index + tile_y + 4+ (tile_x + 2)*F]= temp24;
				xx[index + tile_y + 5+ (tile_x + 2)*F]= temp25;
				xx[index + tile_y + 6+ (tile_x + 2)*F]= temp26;
				xx[index + tile_y + 7+ (tile_x + 2)*F]= temp27;
				xx[index + tile_y + 8+ (tile_x + 2)*F]= temp28;
				xx[index + tile_y + 9+ (tile_x + 2)*F]= temp29;


				xx[index + tile_y + 0+ (tile_x + 3)*F]= temp30;
				xx[index + tile_y + 1+ (tile_x + 3)*F]= temp31;
				xx[index + tile_y + 2+ (tile_x + 3)*F]= temp32;
				xx[index + tile_y + 3+ (tile_x + 3)*F]= temp33;
				xx[index + tile_y + 4+ (tile_x + 3)*F]= temp34;
				xx[index + tile_y + 5+ (tile_x + 3)*F]= temp35;
				xx[index + tile_y + 6+ (tile_x + 3)*F]= temp36;
				xx[index + tile_y + 7+ (tile_x + 3)*F]= temp37;
				xx[index + tile_y + 8+ (tile_x + 3)*F]= temp38;
				xx[index + tile_y + 9+ (tile_x + 3)*F]= temp39;


				xx[index + tile_y + 0+ (tile_x + 4)*F]= temp40;
				xx[index + tile_y + 1+ (tile_x + 4)*F]= temp41;
				xx[index + tile_y + 2+ (tile_x + 4)*F]= temp42;
				xx[index + tile_y + 3+ (tile_x + 4)*F]= temp43;
				xx[index + tile_y + 4+ (tile_x + 4)*F]= temp44;
				xx[index + tile_y + 5+ (tile_x + 4)*F]= temp45;
				xx[index + tile_y + 6+ (tile_x + 4)*F]= temp46;
				xx[index + tile_y + 7+ (tile_x + 4)*F]= temp47;
				xx[index + tile_y + 8+ (tile_x + 4)*F]= temp48;
				xx[index + tile_y + 9+ (tile_x + 4)*F]= temp49;


				xx[index + tile_y + 0+ (tile_x + 5)*F]= temp50;
				xx[index + tile_y + 1+ (tile_x + 5)*F]= temp51;
				xx[index + tile_y + 2+ (tile_x + 5)*F]= temp52;
				xx[index + tile_y + 3+ (tile_x + 5)*F]= temp53;
				xx[index + tile_y + 4+ (tile_x + 5)*F]= temp54;
				xx[index + tile_y + 5+ (tile_x + 5)*F]= temp55;
				xx[index + tile_y + 6+ (tile_x + 5)*F]= temp56;
				xx[index + tile_y + 7+ (tile_x + 5)*F]= temp57;
				xx[index + tile_y + 8+ (tile_x + 5)*F]= temp58;
				xx[index + tile_y + 9+ (tile_x + 5)*F]= temp59;


				xx[index + tile_y + 0+ (tile_x + 6)*F]= temp60;
				xx[index + tile_y + 1+ (tile_x + 6)*F]= temp61;
				xx[index + tile_y + 2+ (tile_x + 6)*F]= temp62;
				xx[index + tile_y + 3+ (tile_x + 6)*F]= temp63;
				xx[index + tile_y + 4+ (tile_x + 6)*F]= temp64;
				xx[index + tile_y + 5+ (tile_x + 6)*F]= temp65;
				xx[index + tile_y + 6+ (tile_x + 6)*F]= temp66;
				xx[index + tile_y + 7+ (tile_x + 6)*F]= temp67;
				xx[index + tile_y + 8+ (tile_x + 6)*F]= temp68;
				xx[index + tile_y + 9+ (tile_x + 6)*F]= temp69;


				xx[index + tile_y + 0+ (tile_x + 7)*F]= temp70;
				xx[index + tile_y + 1+ (tile_x + 7)*F]= temp71;
				xx[index + tile_y + 2+ (tile_x + 7)*F]= temp72;
				xx[index + tile_y + 3+ (tile_x + 7)*F]= temp73;
				xx[index + tile_y + 4+ (tile_x + 7)*F]= temp74;
				xx[index + tile_y + 5+ (tile_x + 7)*F]= temp75;
				xx[index + tile_y + 6+ (tile_x + 7)*F]= temp76;
				xx[index + tile_y + 7+ (tile_x + 7)*F]= temp77;
				xx[index + tile_y + 8+ (tile_x + 7)*F]= temp78;
				xx[index + tile_y + 9+ (tile_x + 7)*F]= temp79;


				xx[index + tile_y + 0+ (tile_x + 8)*F]= temp80;
				xx[index + tile_y + 1+ (tile_x + 8)*F]= temp81;
				xx[index + tile_y + 2+ (tile_x + 8)*F]= temp82;
				xx[index + tile_y + 3+ (tile_x + 8)*F]= temp83;
				xx[index + tile_y + 4+ (tile_x + 8)*F]= temp84;
				xx[index + tile_y + 5+ (tile_x + 8)*F]= temp85;
				xx[index + tile_y + 6+ (tile_x + 8)*F]= temp86;
				xx[index + tile_y + 7+ (tile_x + 8)*F]= temp87;
				xx[index + tile_y + 8+ (tile_x + 8)*F]= temp88;
				xx[index + tile_y + 9+ (tile_x + 8)*F]= temp89;


				xx[index + tile_y + 0+ (tile_x + 9)*F]= temp90;
				xx[index + tile_y + 1+ (tile_x + 9)*F]= temp91;
				xx[index + tile_y + 2+ (tile_x + 9)*F]= temp92;
				xx[index + tile_y + 3+ (tile_x + 9)*F]= temp93;
				xx[index + tile_y + 4+ (tile_x + 9)*F]= temp94;
				xx[index + tile_y + 5+ (tile_x + 9)*F]= temp95;
				xx[index + tile_y + 6+ (tile_x + 9)*F]= temp96;
				xx[index + tile_y + 7+ (tile_x + 9)*F]= temp97;
				xx[index + tile_y + 8+ (tile_x + 9)*F]= temp98;
				xx[index + tile_y + 9+ (tile_x + 9)*F]= temp99;
			}
			//regularization
			if(tile_x == tile_y){
				for(int k = 0; k < tile; k++)
					xx[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}


//*/

	}
}



__global__ void
__launch_bounds__(100, 4)
updateXByBlock2pRegDsmemTile(float* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda) {
	__shared__ float2 thetaTemp[SCAN_BATCH * F/2];
	int row = blockIdx.x;
	if (row < M) {
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
		float2 theta;
		int tile = F/10;
		int tile_x = (threadIdx.x/tile) * tile;//start x of this tile
		int tile_y = (threadIdx.x%tile) * tile;//start y of this tile
		for (int iter = 0; iter < iterations; iter ++){
			//copy texture --> smem, and sync
			if(threadIdx.x < SCAN_BATCH){
				if(iter*SCAN_BATCH + threadIdx.x < end - start){
					for (int k = 0; k < F; k += 2){
						theta.x =
								tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + threadIdx.x] + k);
						theta.y =
								tex1Dfetch(thetaTTexRef, F * csrColIndex[start + iter*SCAN_BATCH + threadIdx.x] + k+1);
						thetaTemp[threadIdx.x * F/2 + k/2] = theta;
					}
				}
				//must be the last iteration; no need to check
				//not enough theta to copy, set zero
				else
					memset(&thetaTemp[threadIdx.x*F/2], 0, F*sizeof(float));
			}
			__syncthreads();
			///////////////////////////////////////////////////////////////////////////////////////////////////////////
			//tile: 10*10
			for(int k = 0; k < SCAN_BATCH; k++){
				temp0 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
				temp1 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
				temp2 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp3 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp4 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp5 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp6 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp7 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp8 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp9 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp10 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
				temp11 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
				temp12 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp13 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp14 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp15 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp16 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp17 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp18 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp19 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp20 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
				temp21 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
				temp22 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp23 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp24 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp25 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp26 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp27 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp28 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp29 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp30 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
				temp31 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
				temp32 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp33 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp34 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp35 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp36 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp37 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp38 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp39 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp40 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
				temp41 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
				temp42 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp43 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp44 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp45 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp46 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp47 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp48 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp49 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp50 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
				temp51 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
				temp52 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp53 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp54 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp55 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp56 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp57 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp58 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp59 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp60 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
				temp61 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
				temp62 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp63 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp64 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp65 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp66 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp67 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp68 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp69 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp70 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
				temp71 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
				temp72 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp73 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp74 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp75 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp76 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp77 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp78 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp79 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;


				temp80 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
				temp81 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
				temp82 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp83 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp84 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp85 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp86 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp87 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp88 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp89 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

				temp90 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
				temp91 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
				temp92 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
				temp93 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
				temp94 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
				temp95 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
				temp96 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
				temp97 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
				temp98 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
				temp99 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;
			}
			///////////////////////////////////////////////////////////////////////////////////////////////////////////
			__syncthreads();
		}
		int index = blockIdx.x*F*F;
///*
		//copy output to gmem
		tt[index + tile_x + tile_y*F] = temp0;
		tt[index + tile_x + (tile_y + 1)*F] = temp1;
		tt[index + tile_x + (tile_y + 2)*F] = temp2;
		tt[index + tile_x + (tile_y + 3)*F] = temp3;
		tt[index + tile_x + (tile_y + 4)*F] = temp4;
		tt[index + tile_x + (tile_y + 5)*F] = temp5;
		tt[index + tile_x + (tile_y + 6)*F] = temp6;
		tt[index + tile_x + (tile_y + 7)*F] = temp7;
		tt[index + tile_x + (tile_y + 8)*F] = temp8;
		tt[index + tile_x + (tile_y + 9)*F] = temp9;

		tt[index + tile_x + 1 + tile_y*F] = temp10;
		tt[index + tile_x + 1 + (tile_y + 1)*F] = temp11;
		tt[index + tile_x + 1 + (tile_y + 2)*F] = temp12;
		tt[index + tile_x + 1 + (tile_y + 3)*F] = temp13;
		tt[index + tile_x + 1 + (tile_y + 4)*F] = temp14;
		tt[index + tile_x + 1 + (tile_y + 5)*F] = temp15;
		tt[index + tile_x + 1 + (tile_y + 6)*F] = temp16;
		tt[index + tile_x + 1 + (tile_y + 7)*F] = temp17;
		tt[index + tile_x + 1 + (tile_y + 8)*F] = temp18;
		tt[index + tile_x + 1 + (tile_y + 9)*F] = temp19;

		tt[index + tile_x + 2 + tile_y*F] = temp20;
		tt[index + tile_x + 2 + (tile_y + 1)*F] = temp21;
		tt[index + tile_x + 2 + (tile_y + 2)*F] = temp22;
		tt[index + tile_x + 2 + (tile_y + 3)*F] = temp23;
		tt[index + tile_x + 2 + (tile_y + 4)*F] = temp24;
		tt[index + tile_x + 2 + (tile_y + 5)*F] = temp25;
		tt[index + tile_x + 2 + (tile_y + 6)*F] = temp26;
		tt[index + tile_x + 2 + (tile_y + 7)*F] = temp27;
		tt[index + tile_x + 2 + (tile_y + 8)*F] = temp28;
		tt[index + tile_x + 2 + (tile_y + 9)*F] = temp29;

		tt[index + tile_x + 3 + tile_y*F] = temp30;
		tt[index + tile_x + 3 + (tile_y + 1)*F] = temp31;
		tt[index + tile_x + 3 + (tile_y + 2)*F] = temp32;
		tt[index + tile_x + 3 + (tile_y + 3)*F] = temp33;
		tt[index + tile_x + 3 + (tile_y + 4)*F] = temp34;
		tt[index + tile_x + 3 + (tile_y + 5)*F] = temp35;
		tt[index + tile_x + 3 + (tile_y + 6)*F] = temp36;
		tt[index + tile_x + 3 + (tile_y + 7)*F] = temp37;
		tt[index + tile_x + 3 + (tile_y + 8)*F] = temp38;
		tt[index + tile_x + 3 + (tile_y + 9)*F] = temp39;

		tt[index + tile_x + 4 + tile_y*F] = temp40;
		tt[index + tile_x + 4 + (tile_y + 1)*F] = temp41;
		tt[index + tile_x + 4 + (tile_y + 2)*F] = temp42;
		tt[index + tile_x + 4 + (tile_y + 3)*F] = temp43;
		tt[index + tile_x + 4 + (tile_y + 4)*F] = temp44;
		tt[index + tile_x + 4 + (tile_y + 5)*F] = temp45;
		tt[index + tile_x + 4 + (tile_y + 6)*F] = temp46;
		tt[index + tile_x + 4 + (tile_y + 7)*F] = temp47;
		tt[index + tile_x + 4 + (tile_y + 8)*F] = temp48;
		tt[index + tile_x + 4 + (tile_y + 9)*F] = temp49;

		tt[index + tile_x + 5 + tile_y*F] = temp50;
		tt[index + tile_x + 5 + (tile_y + 1)*F] = temp51;
		tt[index + tile_x + 5 + (tile_y + 2)*F] = temp52;
		tt[index + tile_x + 5 + (tile_y + 3)*F] = temp53;
		tt[index + tile_x + 5 + (tile_y + 4)*F] = temp54;
		tt[index + tile_x + 5 + (tile_y + 5)*F] = temp55;
		tt[index + tile_x + 5 + (tile_y + 6)*F] = temp56;
		tt[index + tile_x + 5 + (tile_y + 7)*F] = temp57;
		tt[index + tile_x + 5 + (tile_y + 8)*F] = temp58;
		tt[index + tile_x + 5 + (tile_y + 9)*F] = temp59;

		tt[index + tile_x + 6 + tile_y*F] = temp60;
		tt[index + tile_x + 6 + (tile_y + 1)*F] = temp61;
		tt[index + tile_x + 6 + (tile_y + 2)*F] = temp62;
		tt[index + tile_x + 6 + (tile_y + 3)*F] = temp63;
		tt[index + tile_x + 6 + (tile_y + 4)*F] = temp64;
		tt[index + tile_x + 6 + (tile_y + 5)*F] = temp65;
		tt[index + tile_x + 6 + (tile_y + 6)*F] = temp66;
		tt[index + tile_x + 6 + (tile_y + 7)*F] = temp67;
		tt[index + tile_x + 6 + (tile_y + 8)*F] = temp68;
		tt[index + tile_x + 6 + (tile_y + 9)*F] = temp69;

		tt[index + tile_x + 7 + tile_y*F] = temp70;
		tt[index + tile_x + 7 + (tile_y + 1)*F] = temp71;
		tt[index + tile_x + 7 + (tile_y + 2)*F] = temp72;
		tt[index + tile_x + 7 + (tile_y + 3)*F] = temp73;
		tt[index + tile_x + 7 + (tile_y + 4)*F] = temp74;
		tt[index + tile_x + 7 + (tile_y + 5)*F] = temp75;
		tt[index + tile_x + 7 + (tile_y + 6)*F] = temp76;
		tt[index + tile_x + 7 + (tile_y + 7)*F] = temp77;
		tt[index + tile_x + 7 + (tile_y + 8)*F] = temp78;
		tt[index + tile_x + 7 + (tile_y + 9)*F] = temp79;

		tt[index + tile_x + 8 + tile_y*F] = temp80;
		tt[index + tile_x + 8 + (tile_y + 1)*F] = temp81;
		tt[index + tile_x + 8 + (tile_y + 2)*F] = temp82;
		tt[index + tile_x + 8 + (tile_y + 3)*F] = temp83;
		tt[index + tile_x + 8 + (tile_y + 4)*F] = temp84;
		tt[index + tile_x + 8 + (tile_y + 5)*F] = temp85;
		tt[index + tile_x + 8 + (tile_y + 6)*F] = temp86;
		tt[index + tile_x + 8 + (tile_y + 7)*F] = temp87;
		tt[index + tile_x + 8 + (tile_y + 8)*F] = temp88;
		tt[index + tile_x + 8 + (tile_y + 9)*F] = temp89;

		tt[index + tile_x + 9 + tile_y*F] = temp90;
		tt[index + tile_x + 9 + (tile_y + 1)*F] = temp91;
		tt[index + tile_x + 9 + (tile_y + 2)*F] = temp92;
		tt[index + tile_x + 9 + (tile_y + 3)*F] = temp93;
		tt[index + tile_x + 9 + (tile_y + 4)*F] = temp94;
		tt[index + tile_x + 9 + (tile_y + 5)*F] = temp95;
		tt[index + tile_x + 9 + (tile_y + 6)*F] = temp96;
		tt[index + tile_x + 9 + (tile_y + 7)*F] = temp97;
		tt[index + tile_x + 9 + (tile_y + 8)*F] = temp98;
		tt[index + tile_x + 9 + (tile_y + 9)*F] = temp99;
//*/
		//regularization
		if(tile_x == tile_y){
			for(int k = 0; k < tile; k++)
				tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
		}
	}

}

void loadCSRSparseMatrix(const char* dataFile, const char* rowFile, const char* colFile,
		float* data, unsigned int* row, int* col) {
    printf("\n loading CSR...\n");
	FILE *dFile = fopen(dataFile,"rb");
	FILE *rFile = fopen(rowFile,"rb");
	FILE *cFile = fopen(colFile,"rb");
	if (!rFile||!dFile||!cFile)
	{
		printf("Unable to open file!");
		return;
	}

	fread(&row[0], 4*(M+1) ,1, rFile);
	fread(&col[0], 4*NNZ ,1, cFile);
	fread(&data[0], 4*NNZ ,1, dFile);

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
}

void loadCSCSparseMatrix(const char* dataFile, const char* rowFile, const char* colFile, float * data, int* row, int* col) {
    printf("\n loading CSC...\n");

	FILE *dFile = fopen(dataFile,"rb");
	FILE *rFile = fopen(rowFile,"rb");
	FILE *cFile = fopen(colFile,"rb");
	if (!rFile||!dFile||!dFile)
	{
		printf("Unable to open file!");
		return;
	}

	fread(&data[0], 4*NNZ ,1, dFile);
	fread(&row[0], 4*NNZ ,1, rFile);
	fread(&col[0], 4*(N+1) ,1, cFile);

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
}
void loadCSCSparseMatrixInBatch(const std::string dataFile, const std::string rowFile, const std::string colFile, float * data, int* row, int* col, long csc_nnz, int n) {
    printf("\n loading CSC from %s, %s, %s \n", dataFile.c_str(), rowFile.c_str(), colFile.c_str());

	FILE *dFile = fopen(dataFile.c_str(),"rb");
	FILE *rFile = fopen(rowFile.c_str(),"rb");
	FILE *cFile = fopen(colFile.c_str(),"rb");
	if (!rFile||!dFile||!dFile)
	{
		printf("Unable to open file!");
		return;
	}

	fread(&data[0], 4*csc_nnz ,1, dFile);
	fread(&row[0], 4*csc_nnz ,1, rFile);
	fread(&col[0], 4*(n+1) ,1, cFile);

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
}

void loadCooSparseMatrixRowPtr(const char* rowFile, int* row) {
    printf("\n loading COO...\n");
	FILE *rfile = fopen(rowFile,"rb");
	fread(&row[0], 4*NNZ ,1, rfile);
	fclose(rfile);
	//FILE *file = fopen("./hugewiki_R_train_coo.row.bin", "wb");
	//fwrite(row, 4*NNZ, 1, file);
	//fclose(file);

}

void loadCooSparseMatrix(const char* dataFile, const char* rowFile, const char* colFile,
		float* data, int* row, int* col, int nnz) {
	std::ifstream dfile(dataFile);
	std::ifstream rfile(rowFile);
	std::ifstream cfile(colFile);

	float d;
	int d_i = 0;
	while (dfile >> d) {
//printf("%f ",d);
		data[d_i++] = d;
	}

	int r;
	int r_i = 0;
	while (rfile >> r) {
//printf("%d ",r);
		row[r_i++] = r;
	}
	int c;
	int c_i = 0;
	while (cfile >> c) {
//printf("%d ",c);
		col[c_i++] = c;
	}
}

inline void updateX(const int batch_id, const int batch_size, const long batch_offset, float * ythetaT, float * tt, float * XT_h,
		cublasHandle_t handle, const int m, const int n, const int f, const int nnz,
		float** devPtrTTHost, float **devPtrYthetaTHost,
		float **devPtrTT, float **devPtrYthetaT, int *P, int *INFO){
	double t0 = seconds();
	//left-hand side pointers
	for (int k = 0; k < batch_size; k++) {
		devPtrTTHost[k] = &tt[k * F * F];
	}
	cudacall(cudaMemcpy(devPtrTT, devPtrTTHost,
			batch_size * sizeof(*devPtrTT),cudaMemcpyHostToDevice));
	int * info2 = (int *) malloc(sizeof(int));
	//right-hand side pointer
	for (int k = 0; k < batch_size; k++) {
		devPtrYthetaTHost[k] = &ythetaT[k * F];
	}
	cudacall(cudaMemcpy(devPtrYthetaT, devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaT),
			cudaMemcpyHostToDevice));

	//getrf then getrs
    //printf("\t\t\tbatch %d, prepare in secs: %f\n", batch_id, seconds() - t0);
	//t0 = seconds();
	cublasSgetrfBatched(handle, F, devPtrTT, F, P, INFO, batch_size);
	//cudaDeviceSynchronize();
    //cudaCheckError();
    //printf("\t\t\tbatch %d, LU factorization of tt in secs: %f\n", batch_id, seconds() - t0);

	//t0 = seconds();
	cublasSgetrsBatched(handle, CUBLAS_OP_N, F, 1,
			(const float ** ) devPtrTT, F, P, devPtrYthetaT, F, info2, batch_size);
    //cudaDeviceSynchronize();
    //cudaCheckError();
    //printf("\t\t\tbatch %d, solve after LU in secs: %f\n", batch_id, seconds() - t0);
	//t0 = seconds();
    cudacall( cudaMemcpy(&XT_h[batch_offset * F], ythetaT,
			batch_size * F * sizeof(float), cudaMemcpyDeviceToHost) );
    //printf("\t\t\tbatch %d, copy to host XT_h secs: %f\n", batch_id, seconds() - t0);
}

int updateTheta(const int batch_size, const int batch_offset, float * xx,
		  float * yTXT, float * thetaT,
		cublasHandle_t handle, const int n, const int f){

	float ** devPtrXXHost = (float**) malloc(batch_size * sizeof(devPtrXXHost[0]));
	float **devPtrXX = 0;

	for (int k = 0; k < batch_size; k++) {
		devPtrXXHost[k] = &xx[k * F * F];
	}
	cudacall(cudaMalloc((void** ) &devPtrXX, batch_size * sizeof(*devPtrXX)));
	cudacall(cudaMemcpy(devPtrXX, devPtrXXHost, batch_size * sizeof(*devPtrXX), cudaMemcpyHostToDevice));
	int *P, *INFO;
	cudacall(cudaMalloc(&P, f * batch_size * sizeof(int)));
	cudacall(cudaMalloc(&INFO, batch_size * sizeof(int)));
	cublasSgetrfBatched(handle, F, devPtrXX, F, P, INFO, batch_size);
    cudaDeviceSynchronize();
    cudaCheckError();

	//gettimeofday(&tv1, NULL);
	//elapsed = (tv1.tv_sec - tv0.tv_sec)
	//		+ (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
	//printf("\t %f seconds. \n", elapsed);



	//printf("******* solve xx * thetaT = yTXT with CUDA 7.\n");

	float **devPtrYTXTHost = 0;
	float **devPtrYTXT = 0;
	devPtrYTXTHost = (float**) malloc(batch_size * sizeof(devPtrYTXTHost[0]));

	for (int k = 0; k < batch_size; k++) {
		devPtrYTXTHost[k] = &yTXT[k * F];
	}

	cudacall(cudaMalloc((void** ) &devPtrYTXT, batch_size * sizeof(*devPtrYTXT)));
	cudacall(cudaMemcpy(devPtrYTXT, devPtrYTXTHost, batch_size * sizeof(*devPtrYTXT),cudaMemcpyHostToDevice));

	int * info2 = (int *) malloc(sizeof(int));
	cublasSgetrsBatched(handle, CUBLAS_OP_N, F, 1,
			(const float ** ) devPtrXX, F, P, devPtrYTXT, F, info2, batch_size);
    cudaDeviceSynchronize();
    cudaCheckError();
	cudacall( cudaMemcpy( &thetaT[batch_offset * F], yTXT,
			batch_size * F * sizeof(float), cudaMemcpyDeviceToDevice) );

	//gettimeofday(&tv2, NULL);
	//elapsed = (tv2.tv_sec - tv1.tv_sec)
	//		+ (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
	//printf("\t %f seconds. \n", elapsed);
	/*
	//testing purpose

	float* yTXHost = (float *) malloc(f * n * sizeof(yTXHost[0]));

	cudacall(cudaMemcpy(yTXHost, yTXT, n * f * sizeof(float), cudaMemcpyDeviceToHost));
	printf("\n*********yTXT***\n");
	for (int i = 0; i < n * f; i++) {
		printf("%f\t", yTXHost[i]);
	}
	printf("\n");
	 */
	/*
	float* thetaTHost = (float *) malloc(f * n * sizeof(thetaTHost[0]));

	cudacall( cudaMemcpy(thetaTHost, thetaT, n * f * sizeof(float),cudaMemcpyDeviceToHost));
	printf("\n*********ThetaT***\n");
	for (int i = 0; i < n * f; i++) {
		printf("%f\t", thetaTHost[i]);
	}
	printf("\n");
	 */
	free(devPtrXXHost);
	cudaFree(devPtrXX);
	cudaFree(P);
	cudaFree(INFO);
	free(info2);

	free(devPtrYTXTHost);
	cudaFree(devPtrYTXT);

	return 0;
}

__global__ void RMSE(const float * csrVal, const int* cooRowIndex,
		const int* csrColIndex, const float * thetaT, const float * XT, float * error, const int nnz,
		const int error_size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nnz) {
		int row = cooRowIndex[i];
		int col = csrColIndex[i];
		float e = csrVal[i];
		//if(i%1000000==0) printf("row: %d, col: %d, csrVal[%d]: %f.\t", row, col, i, e);
		for (int k = 0; k < F; k++) {
			e -= tex1Dfetch(thetaTTexRef, F * col + k) * tex1Dfetch(xTTexRef, F * row + k);
		}
		atomicAdd(&error[i%error_size], e*e);
		//error[i] = e*e;
		//if(i%1000000==0) printf("error[%d]: %f.\n", i, e);
	}
}

__global__ void RMSE_CSC(const float * cscVal, const int* cscRowIndex,
		const int* cscColIndex, const float * thetaT, const float * XT, float * error,
		const int error_size, int* nan) {
	int col = blockIdx.x;
	int start = cscColIndex[col];
	int end = cscColIndex[col + 1];
	if (col < N && threadIdx.x < end - start) {
		for (int i = 0; threadIdx.x + i*blockDim.x < end - start; i++) {
			int index = start + i*blockDim.x + threadIdx.x;
			float e0 = cscVal[index];
			float e = e0;
			//if(isnan(e)) printf("ERROR: NAN***\n");
			int row = cscRowIndex[index];
			//if(isfinite(((double)row))) printf("ERROR: NAN@@@\n");
			for (int k = 0; k < F; k++) {

				e -= tex1Dfetch(thetaTTexRef, F * col + k) * XT[ F * row  + k];
				//TODO: fix this, a user/item does not show up in training

				//if(isnan(e1)) printf("e1: NAN!!!%d, %d, %d\n", index, col, row);
				//if(isnan(e2)) printf("e2: NAN!!!%d, %d, %d\n", index, col, row);


			}
			if(isnan(e)) {
				e = 0;
				atomicAdd(&nan[0],1);
			}
			//if(isnan(e)) printf("ERROR: NAN!!!%d, %d, %d\n", index, col, row);
			atomicAdd(&error[row%error_size], e*e);
		}
	}
}

int main() {
	printf("enable p2p among %d GPUs if available.\n", GPU_COUNT);
	enableP2P(GPU_COUNT);
	//initialize cublas, cusparse
	cublasHandle_t handle[GPU_COUNT];
	cusparseHandle_t cushandle[GPU_COUNT];
	for(int gpu_id = 0; gpu_id < GPU_COUNT; gpu_id ++){
		cudacall(cudaSetDevice(gpu_id));
		cublascall(cublasCreate(&handle[gpu_id]));
		cusparsecall(cusparseCreate(&cushandle[gpu_id]));
	}
	cudaSetDevice(DEVICEID);
	long m = M;
	long n = N;
	long f = F;
	long nnz = NNZ;
	float lambda = LAMBDA;

	unsigned int* csrRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrRowIndexHostPtr, (m + 1) * sizeof(int)) );
	int* csrColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrColIndexHostPtr, nnz * sizeof(int)) );
	float* csrValHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrValHostPtr, nnz * sizeof(float)) );

	long csc_nnz[GPU_COUNT] = {777607310, 773335400, 777305655, 772895948};
	long csc_m[GPU_COUNT] = {12520650, 12520650, 12520650, 12520653};
	long csc_nnz_test[GPU_COUNT] = {86418516, 85913272, 86357875, 85883667};

	float* cscValHostPtr[GPU_COUNT];
	int* cscRowIndexHostPtr[GPU_COUNT];
	int* cscColIndexHostPtr[GPU_COUNT];
	for(int gpu_id = 0; gpu_id < GPU_COUNT; gpu_id ++){
		cudacall(cudaMallocHost( (void** ) &cscValHostPtr[gpu_id], csc_nnz[gpu_id] * sizeof(float)) );
		cudacall(cudaMallocHost( (void** ) &cscRowIndexHostPtr[gpu_id], csc_nnz[gpu_id] * sizeof(int)) );
		cudacall(cudaMallocHost( (void** ) &cscColIndexHostPtr[gpu_id], (n+1) * sizeof(int)) );
	}

	float* testCscValHostPtr[GPU_COUNT];
	int* testCscRowIndexHostPtr[GPU_COUNT];
	int* testCscColIndexHostPtr[GPU_COUNT];
	for(int gpu_id = 0; gpu_id < GPU_COUNT; gpu_id ++){
		cudacall(cudaMallocHost( (void** ) &testCscValHostPtr[gpu_id], csc_nnz_test[gpu_id] * sizeof(float)) );
		cudacall(cudaMallocHost( (void** ) &testCscRowIndexHostPtr[gpu_id], csc_nnz_test[gpu_id] * sizeof(int)) );
		cudacall(cudaMallocHost( (void** ) &testCscColIndexHostPtr[gpu_id], (n+1) * sizeof(int)) );
	}

	//calculate X from thetaT first, need to initialize thetaT
	float* thetaTHost;
	cudacall(cudaMallocHost( (void** ) &thetaTHost, n * f * sizeof(float)) );

	//index of XT_h need a long -- beyond what int32 can handle (2^31 or 2^32)
	float * XT_h;
	//cudacall (cudaHostAlloc((void **)&XT_h,  f * m * sizeof(XT_h[0]), cudaHostAllocMapped) );
	cudacall (cudaMallocHost((void **)&XT_h,  f * m * sizeof(XT_h[0])) );
	//initialize thetaT on host
	srand (time(0));
	for (int k = 0; k < n * f; k++)
		thetaTHost[k] = 0.5*((float) rand() / (RAND_MAX));
		//thetaTHost[k] = 0.1*((float) rand() / (float)RAND_MAX);
		//thetaTHost[k] = 0;
	//CG needs an initial value of XT
	memset(XT_h,0,m*f*sizeof(float));
	//for (long k = 0; k < m * f; k++)
	//	XT_h[k] = 0.5*((float) rand() / (RAND_MAX));
	//device pointers
	int * csrRowIndex[GPU_COUNT];
	int * csrColIndex[GPU_COUNT];
	float * csrVal[GPU_COUNT];
	float * thetaT[GPU_COUNT];
	float * XT_d[GPU_COUNT];

	float * cscVal[GPU_COUNT];
	int * cscRowIndex[GPU_COUNT];
	int * cscColIndex[GPU_COUNT];

	printf("*******starting loading training and testing sets to host.\n");
    loadCSRSparseMatrix("../data/hugewiki/hugewiki_R_train_csr.data", "../data/hugewiki/hugewiki_R_train_csr.indptr", "../data/hugewiki/hugewiki_R_train_csr.indices",
    		csrValHostPtr, csrRowIndexHostPtr, csrColIndexHostPtr);

	omp_set_num_threads(GPU_COUNT);
#pragma omp parallel
	{
		int gpu_id = omp_get_thread_num();
		std::string str1("../data/hugewiki/hugewiki_R_train_csc.data.bin");
		std::string str2("../data/hugewiki/hugewiki_R_train_csc.indices.bin");
		std::string str3("../data/hugewiki/hugewiki_R_train_csc.indptr.bin");

		//printf("%s",(str+to_string(gpu_id)).c_str());
	    loadCSCSparseMatrixInBatch((str1 + to_string(gpu_id)).c_str(),
	    		(str2 + to_string(gpu_id)).c_str(),
	    		(str3 + to_string(gpu_id)).c_str(),
	    		cscValHostPtr[gpu_id], cscRowIndexHostPtr[gpu_id], cscColIndexHostPtr[gpu_id], csc_nnz[gpu_id], n);
	}
	#pragma omp parallel
	{
		int gpu_id = omp_get_thread_num();
		std::string str1("../data/hugewiki/hugewiki_R_test_csc.data.bin");
		std::string str2("../data/hugewiki/hugewiki_R_test_csc.indices.bin");
		std::string str3("../data/hugewiki/hugewiki_R_test_csc.indptr.bin");

		//printf("%s",(str+to_string(gpu_id)).c_str());
		loadCSCSparseMatrixInBatch((str1 + to_string(gpu_id)).c_str(),
				(str2 + to_string(gpu_id)).c_str(),
				(str3 + to_string(gpu_id)).c_str(),
				testCscValHostPtr[gpu_id], testCscRowIndexHostPtr[gpu_id],
				testCscColIndexHostPtr[gpu_id], csc_nnz_test[gpu_id], n);
	}

    printf("\n loaded csr to host; print data, row and col array\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%f ", csrValHostPtr[i]);
	}
	printf("\n");

	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", csrRowIndexHostPtr[i]);
	}
	printf("\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", csrColIndexHostPtr[i]);
	}
	printf("\n");

    printf("\n loaded csc to host; print data, row and col array\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%f ", cscValHostPtr[0][i]);
	}
	printf("\n");

	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", cscRowIndexHostPtr[0][i]);
	}
	printf("\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", cscColIndexHostPtr[0][i]);
	}
	printf("\n");

    printf("\n loaded csc test to host; print data, row and col array\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%f ", testCscValHostPtr[0][i]);
	}
	printf("\n");

	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", testCscRowIndexHostPtr[0][i]);
	}
	printf("\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", testCscColIndexHostPtr[0][i]);
	}
	printf("\n");

	cudacall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	//64-bit smem access
	//http://acceleware.com/blog/maximizing-shared-memory-bandwidth-nvidia-kepler-gpus
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	cudaSharedMemConfig  pConfig;
	cudaDeviceGetSharedMemConfig (&pConfig);
	//printf("%d\n", pConfig);


	cudacall(cudaSetDevice(DEVICEID));
	cusparseMatDescr_t descr;
	cusparsecall( cusparseCreateMatDescr(&descr));
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	using namespace std;

	//variable used to time
	double t0;
	double elapsed = 0.0;
	struct timeval tv;
	struct timeval start_tv;

	const float alpha = 1.0f;
	const float beta = 0.0f;
	for(int gpu_id = 0; gpu_id < GPU_COUNT; gpu_id ++){
        cudacall(cudaSetDevice(gpu_id));
		cudacall(cudaMalloc((void** ) &thetaT[gpu_id], f * n * sizeof(float)));
		printf("*******copy memory to GPU %d...\n", gpu_id);
		cudacall(cudaMemcpy(thetaT[gpu_id], thetaTHost, (size_t ) (n * f * sizeof(float)), cudaMemcpyHostToDevice));
	}

	//host pointers for cublas batch operations
	float ** devPtrTTHost[GPU_COUNT];
	float **devPtrYthetaTHost[GPU_COUNT];
	for(int iter = 0; iter < ITERS ; iter ++){
		printf("---------------------------update X iteration %d ----------------------------------\n", iter);
		t0 = seconds();
		//parallel in all GPUs, or only 1
		int parallelism_level = GPU_COUNT;
		omp_set_num_threads(parallelism_level);
		//gpu memory to be used across batches
		//last batch size, the largest among batches
		int batch_size_max = m - (X_BATCH - 1)*(m/X_BATCH);

		int counter = 0;
		#pragma omp parallel shared (counter)
		{
			//this is the code on one gpu
			int gpu_id = omp_get_thread_num();
	        cudacall(cudaSetDevice(gpu_id));

	        //for batch solvers
			cudacall(cudaMallocHost( (void** ) &devPtrTTHost[gpu_id], batch_size_max * sizeof(*devPtrTTHost) ) );
			cudacall(cudaMallocHost( (void** ) &devPtrYthetaTHost[gpu_id], batch_size_max * sizeof(*devPtrYthetaTHost) ) );

			float * thetaT_local = thetaT[gpu_id];
			cudacall (cudaBindTexture(NULL, thetaTTexRef, thetaT_local, n * f * sizeof(float)));
			float * tt = 0;
			//last batch size, the largest among batches
			int batch_size = m - (X_BATCH - 1)*(m/X_BATCH);
			//TODO: to get batch_nnz_max from csrRowIndexHostPtr
			int batch_nnz_max = 16000000;
			long batch_offset;

			cudacall(cudaMalloc((void** ) &csrRowIndex[gpu_id],(batch_size + 1) * sizeof(csrRowIndex[0][0])));
			cudacall(cudaMalloc((void** ) &csrColIndex[gpu_id], batch_nnz_max * sizeof(csrColIndex[0][0])));
			cudacall(cudaMalloc((void** ) &csrVal[gpu_id], batch_nnz_max * sizeof(csrVal[0][0])));
			float * ytheta = 0;
			float * ythetaT = 0;
			cudacall(cudaMalloc((void** ) &ytheta, f * batch_size * sizeof(ytheta[0])));
			cudacall(cudaMalloc((void** ) &ythetaT, f * batch_size * sizeof(ythetaT[0])));
			#ifdef CUMF_TT_FP16
			cudacall(cudaMalloc((void** ) &tt, f/2 * f * batch_size * sizeof(float)));
			#else
			cudacall(cudaMalloc((void** ) &tt, f * f * batch_size * sizeof(float)));
			#endif
			//for batch solvers
			float **devPtrTT = 0;
			float **devPtrYthetaT = 0;
			int *P, *INFO;
			cudacall(cudaMalloc((void** ) &devPtrTT, batch_size * sizeof(*devPtrTT)));
			cudacall(cudaMalloc(&P, f * batch_size * sizeof(int)) );
			cudacall(cudaMalloc(&INFO, batch_size * sizeof(int) ));
			cudacall(cudaMalloc((void** ) &devPtrYthetaT, batch_size * sizeof(*devPtrYthetaT)));
			int batch_id = 0;

			//gpu 0 handles batches 0, 4, 8 ...
			//for(int batch_id = gpu_id; batch_id < X_BATCH; batch_id += parallelism_level)
			while(counter < X_BATCH)
			{
				#pragma omp critical
                {
                     batch_id = counter;
                     counter =  counter + 1;
                }

				double t2 = 0;
				t2 = seconds();
				if(batch_id != X_BATCH - 1)
					batch_size = m/X_BATCH;
				batch_offset = batch_id * (m/X_BATCH);
				int batch_nnz =
						csrRowIndexHostPtr[batch_offset + batch_size] - csrRowIndexHostPtr[batch_offset];
				printf("\tbatch %d of %d; size: %d, offset: %d, batch_nnz %d, on gpu %d\n",
						batch_id, X_BATCH, batch_size, batch_offset, batch_nnz, gpu_id);
				//copy CSR rating matrices in
				cudacall(cudaMemcpy(csrRowIndex[gpu_id], &csrRowIndexHostPtr[batch_offset],
						(batch_size + 1) * sizeof(csrRowIndex[0][0]), cudaMemcpyHostToDevice));
				//in place update: csrRowIndex --> csrRowIndex - csrRowIndex[0]
				zeroIndex<<<(batch_size + 1 - 1)/1024 + 1, 1024>>>
						(csrRowIndex[gpu_id], csrRowIndexHostPtr[batch_offset], batch_size + 1);
				cudacall(cudaMemcpy(csrColIndex[gpu_id], &csrColIndexHostPtr[csrRowIndexHostPtr[batch_offset]],
						batch_nnz * sizeof(csrColIndex[0][0]), cudaMemcpyHostToDevice));
				cudacall(cudaMemcpy(csrVal[gpu_id], &csrValHostPtr[csrRowIndexHostPtr[batch_offset]],
						batch_nnz * sizeof(csrVal[0][0]),cudaMemcpyHostToDevice));
				//process right hand: Y*theta
				cusparseScsrmm2(cushandle[gpu_id], CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_TRANSPOSE, batch_size, f, n, batch_nnz, &alpha, descr, csrVal[gpu_id],
						csrRowIndex[gpu_id], csrColIndex[gpu_id], thetaT[gpu_id], f, &beta, ytheta, batch_size);
				//transpose ytheta: ytheta: m*f; need ythetaT = (ytheta).T = f*m
				cublasSgeam(handle[gpu_id], CUBLAS_OP_T, CUBLAS_OP_N, f, batch_size, &alpha,
						(const float * ) ytheta, batch_size, &beta, ythetaT, f, ythetaT, f);
				cudaDeviceSynchronize();
				cudaCheckError();
				//generate left-hand: tt: batch_size*(F*F)
				printf("\t\t batch %d before tt kernel gpu: %d, seconds: %f \n",
						batch_id, gpu_id, seconds() - t2);
				double t1 = seconds();
				#ifdef CUMF_TT_FP16
				get_hermitian100_tt_fp16<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2)>>>
					(0, (half2*) tt, csrRowIndex[gpu_id], csrColIndex[gpu_id], lambda, batch_size, thetaT[gpu_id]);
				#else
				//get_hermitian_x<<<batch_size, 64>>>
				//		(tt, csrRowIndex[gpu_id], csrColIndex[gpu_id], lambda);
				//updateXByBlock2pRegDsmemTile<<<batch_size, F>>>
				//		(tt, csrRowIndex[gpu_id], csrColIndex[gpu_id], lambda);
				get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2)>>>
					(0, tt, csrRowIndex[gpu_id], csrColIndex[gpu_id], lambda, batch_size, thetaT[gpu_id]);
				#endif
				cudaDeviceSynchronize();
				cudaCheckError();
				printf("\t\t batch %d tt kernel gpu: %d, seconds: %f \n",
						batch_id, gpu_id, seconds() - t1);
				t1 = seconds();
/*				
				#ifdef CUMF_SAVE_MODEL
				if(iter==0&&batch_id==0)
					saveDeviceFloatArrayToFile(std::string("../log/0904/hugewiki.tt.hermitkernel"), f * f * batch_size, tt);
				#endif
				updateX(batch_id, batch_size, batch_offset, ythetaT, tt, XT_h,
						handle[gpu_id], m, n, f, nnz, devPtrTTHost[gpu_id], devPtrYthetaTHost[gpu_id],
						devPtrTT, devPtrYthetaT, P, INFO);
				#ifdef CUMF_SAVE_MODEL
				if(iter==0&&batch_id==0)
					saveDeviceFloatArrayToFile(std::string("../log/0904/hugewiki.lu.hermitkernel.xt"), f * batch_size, ythetaT);
				#endif
*/					
///*				
				float * XT = 0;
				cudacall(cudaMalloc((void** ) &XT, f * batch_size * sizeof(XT[0])));
				cudacall( cudaMemcpy(XT, &XT_h[batch_offset * F],
						batch_size * F * sizeof(float), cudaMemcpyHostToDevice) );
				#ifdef CUMF_TT_FP16
				printf("CG solver with fp16.\n");
				updateXWithCGHost_tt_fp16(tt, XT, ythetaT, batch_size, f, 6);
				#else
				printf("CG solver with fp32.\n");
				updateXWithCGHost(tt, XT, ythetaT, batch_size, 100, 100);
				#endif
				cudacall( cudaMemcpy(&XT_h[batch_offset * F], XT,
						batch_size * F * sizeof(float), cudaMemcpyDeviceToHost) );
				#ifdef CUMF_SAVE_MODEL
				if(batch_id==0)
					saveDeviceFloatArrayToFile(std::string("../log/0903/hugewiki.cg.xt.")+ to_string(iter), f * batch_size, XT);
				#endif	
				cudacall(cudaFree(XT));
//*/				

				printf("\t\t batch %d updateX by solving tt , gpu: %d, seconds: %f \n",
						batch_id, gpu_id, seconds() - t1);
				printf("\tbatch %d on gpu %d, runs %f \n", batch_id, gpu_id, seconds() - t2);


			}//end of update x batch

			printf("update X run %f seconds at gpu %d.\n", seconds() - t0, gpu_id);
			cudacall(cudaFree(ytheta));
			cudacall(cudaFree(tt));
			cudacall(cudaFree(csrVal[gpu_id]));
			cudacall(cudaFree(csrRowIndex[gpu_id]));
			cudacall(cudaFree(csrColIndex[gpu_id]));
			cudacall(cudaFree(ythetaT));
			cudaFree(P);
			cudaFree(INFO);
			cudaFree(devPtrTT);
			cudaFree(devPtrYthetaT);
			cudacall(cudaFreeHost(devPtrTTHost[gpu_id]));
			cudacall(cudaFreeHost(devPtrYthetaTHost[gpu_id]));


		}//end of omp parallel loop

		printf("update X run %f seconds, gridSize: %d \n", seconds() -  t0, m);


		gettimeofday(&start_tv, NULL);
		printf("---------------------------------- update theta iteration %d----------------------------------\n",
				iter);
		//in batches, when N is huge
		for(int batch_id = 0; batch_id< THETA_BATCH; batch_id ++){
			int batch_size = 0;
			if(batch_id != THETA_BATCH - 1)
				batch_size = n/THETA_BATCH;
			else
				batch_size = n - batch_id*(n/THETA_BATCH);
			int batch_offset = batch_id * (n/THETA_BATCH);
			printf("batch %d / %d, size: %d\n", batch_id + 1, THETA_BATCH, batch_size);

			float * yTX[GPU_COUNT];
			float * yTXT[GPU_COUNT];

			const float alpha = 1.0f;
			const float beta = 0.0f;
			float * xx[GPU_COUNT];

			omp_set_num_threads(GPU_COUNT);
			t0 = seconds();
#pragma omp parallel
			{
				int gpu_id = omp_get_thread_num();
				long offset = 0;
				for(int k = 0; k < gpu_id; k ++)
					offset += csc_m[k];
				cudacall(cudaSetDevice(gpu_id));
				printf("\tGather xx on GPU %d.\n",gpu_id);
				double t1 = seconds();
				//distribute XT[] to XT_d[i]
				cudacall(cudaMalloc((void** ) &XT_d[gpu_id], f * csc_m[gpu_id] * sizeof(float)));
				//printf("offset: %lld, copy XT_h[%lld] to XT_d[%d]:\n", offset, offset*f, gpu_id);
	        	cudacall(cudaMemcpy(XT_d[gpu_id], &XT_h[offset*f],
	        			f * csc_m[gpu_id] * sizeof(float), cudaMemcpyHostToDevice));
				//copy csc to GPU
				int batch_nnz = cscColIndexHostPtr[gpu_id][batch_offset + batch_size] - cscColIndexHostPtr[gpu_id][batch_offset];
	            cudacall(cudaMalloc((void** ) &cscRowIndex[gpu_id],batch_nnz * sizeof(int)));
	            cudacall(cudaMalloc((void** ) &cscColIndex[gpu_id], (batch_size + 1) * sizeof(int)));
	            cudacall(cudaMalloc((void** ) &cscVal[gpu_id], batch_nnz * sizeof(float)));
	        	cudaMemcpyAsync(cscRowIndex[gpu_id], &cscRowIndexHostPtr[gpu_id][cscColIndexHostPtr[gpu_id][batch_offset]],
	        			batch_nnz * sizeof(cscRowIndex[0][0]), cudaMemcpyHostToDevice);
	        	cudaMemcpy(cscColIndex[gpu_id], &cscColIndexHostPtr[gpu_id][batch_offset],
	        			(batch_size + 1) * sizeof(cscColIndex[0][0]), cudaMemcpyHostToDevice);
				cudaMemcpy(cscVal[gpu_id], &cscValHostPtr[gpu_id][cscColIndexHostPtr[gpu_id][batch_offset]],
				batch_nnz * sizeof(cscVal[0][0]), cudaMemcpyHostToDevice);
				cudacall(cudaMalloc((void** ) &yTXT[gpu_id], f * batch_size * sizeof(float)));
				cudacall(cudaMalloc((void** ) &yTX[gpu_id], f * batch_size * sizeof(float)));
				cudacall(cudaMalloc((void** ) &xx[gpu_id], f * f * batch_size * sizeof(float)));
				printf("\t\tbatch %d memory alloc and cpy gpu %d seconds: %f.\n",
						batch_id, gpu_id, seconds() - t1);


				//in place update: cscColIndex --> cscColIndex - cscColIndex[0]
				zeroIndex<<<(batch_size + 1 - 1)/256 + 1, 256>>>
						(cscColIndex[gpu_id], cscColIndexHostPtr[gpu_id][batch_offset], batch_size + 1);
				//process right-hand side: (Y'*X)'
				cudaDeviceSynchronize();
				cudaCheckError();
				t1 = seconds();
				cusparseScsrmm2(cushandle[gpu_id], CUSPARSE_OPERATION_NON_TRANSPOSE,
						CUSPARSE_OPERATION_TRANSPOSE, batch_size, f, csc_m[gpu_id],
						batch_nnz, &alpha, descr, cscVal[gpu_id], cscColIndex[gpu_id],
						cscRowIndex[gpu_id], XT_d[gpu_id], f, &beta, yTX[gpu_id], batch_size);
				cublasSgeam(handle[gpu_id], CUBLAS_OP_T, CUBLAS_OP_N, f, batch_size, &alpha,
						(const float * ) yTX[gpu_id], batch_size, &beta, yTXT[gpu_id], f, yTXT[gpu_id], f);
				cudaDeviceSynchronize();
				cudaCheckError();
				printf("\t\tbatch %d right-hand side gpu %d seconds: %f.\n", batch_id, gpu_id, seconds() - t1);
				//process left-hand side: generate hessian matrix xx
				t1 = seconds();
				get_hermitian_theta<<<batch_size, 64>>>
						(xx[gpu_id], cscRowIndex[gpu_id], cscColIndex[gpu_id], lambda, XT_d[gpu_id]);
				//get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2)>>>
				//	(0, xx[gpu_id], cscColIndex[gpu_id], cscRowIndex[gpu_id], lambda, batch_size, XT_d[gpu_id]);

				//updateThetaByBlock2pRegDsmemTile<<<batch_size, F>>>
				//		(xx[gpu_id], cscRowIndex[gpu_id], cscColIndex[gpu_id], lambda, XT_d[gpu_id]);
				cudaDeviceSynchronize();
				cudaCheckError();
				printf("\t\tbatch %d xx kernel gpu %d seconds: %f.\n", batch_id, gpu_id, seconds() - t1);
				t1 = seconds();
				cudacall(cudaFree(yTX[gpu_id]));
				cudacall(cudaFree(cscRowIndex[gpu_id]));
				cudacall(cudaFree(cscColIndex[gpu_id]));
				cudacall(cudaFree(cscVal[gpu_id]));
				printf("\t\tbatch %d cudaFree gpu %d seconds: %f.\n", batch_id, gpu_id, seconds() - t1);

			}
			printf("\tbatch %d gather xx in %d GPUs run %f seconds.\n",
					batch_id, GPU_COUNT, seconds() - t0);

			t0 = seconds();
			printf("\t\tadd xx before updateTheta on a given GPU.\n");
			//xx[0] += xx[1] + xx[2] + xx[3]
			cudacall(cudaSetDevice(0));
			float * xx_hotel;
			cudacall(cudaMalloc((void** ) &xx_hotel, f * f * batch_size * sizeof(float)));
			cudaCheckError();

			for(int gpu_id = 1; gpu_id < GPU_COUNT; gpu_id ++){
				//printf("copy from gpu:%d.\n", gpu_id);
				cudacall(cudaMemcpy(xx_hotel, xx[gpu_id], f * f * batch_size * sizeof(float), cudaMemcpyDefault));
				cudaDeviceSynchronize();
				cudaCheckError();
				//printf("add.\n");
				cublasSaxpy(handle[0], f * f * batch_size, &alpha, xx_hotel, 1, xx[0], 1);
				cudaDeviceSynchronize();
				cudaCheckError();
			}
			cudacall(cudaFree(xx_hotel));

			printf("\t\tadd yTXT before updateTheta on a given GPU.\n");
			//xx[0] += xx[1] + xx[2] + xx[3]
			float * yTXT_hotel;
			cudacall(cudaMalloc((void** ) &yTXT_hotel, f * batch_size * sizeof(float)));
			for(int gpu_id = 1; gpu_id < GPU_COUNT; gpu_id ++){
				cudacall(cudaMemcpy(yTXT_hotel, yTXT[gpu_id], f * batch_size * sizeof(float), cudaMemcpyDefault));
				cublasSaxpy(handle[0], f * batch_size, &alpha, yTXT_hotel, 1, yTXT[0], 1);
				cudaDeviceSynchronize();
				cudaCheckError();
			}
			cudacall(cudaFree(yTXT_hotel));
			//printf("*******invoke updateTheta with batch_size: %d, batch_offset: %d.\n", batch_size, batch_offset);
			updateTheta(batch_size, batch_offset, xx[0], yTXT[0], thetaT[0], handle[0], n,  f);
			printf("\tbatch: %d gather and updateTheta in one GPU run %f seconds.\n",
					batch_id, seconds() - t0);

			for(int gpu_id = 0; gpu_id < GPU_COUNT; gpu_id ++){
				cudacall(cudaFree(xx[gpu_id]));
				cudacall(cudaFree(yTXT[gpu_id]));
				cudacall(cudaFree(XT_d[gpu_id]));

			}
		}//end of update theta batches
		//propagate thetaT[0] to non-anchor devices
		for(int gpu_id = 1; gpu_id < GPU_COUNT; gpu_id ++)
			cudacall( cudaMemcpy(thetaT[gpu_id], thetaT[0], n * F * sizeof(float), cudaMemcpyDeviceToDevice) );
		gettimeofday(&tv, NULL);
		elapsed = (tv.tv_sec - start_tv.tv_sec)
				+ (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
		printf("update theta run %f seconds, gridSize: %d.\n", elapsed, n);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		printf("Calculate RMSE in batches.\n");
		//has to calculate in batches since cooRowIndex + csrColIndex + csrVal is so big
		cudacall(cudaSetDevice(0));

		float * errors_train = 0;
		float * errors_test = 0;
		int error_size = 4096;

		int* nan_train = 0;
		int* nan_test = 0;

		cudacall(cudaMalloc((void** ) &errors_train, error_size * sizeof(errors_train[0])));
		cudacall(cudaMemset(errors_train, 0, error_size*sizeof(float)) );

		cudacall(cudaMalloc((void** ) &errors_test, error_size * sizeof(errors_test[0])));
		cudacall(cudaMemset(errors_test, 0, error_size*sizeof(float)) );

		for(int batch_id = 0; batch_id < GPU_COUNT; batch_id ++){
			printf("iteration: %d\n", batch_id);

			int row_offset = 0;
			for(int k = 0; k < batch_id; k ++){
				row_offset += csc_m[k];
			}
			float * XT_small;
			int * cscRowIndex_small;
			int * cscColIndex_small;
			float * cscVal_small;
			cudacall(cudaMalloc((void** ) &XT_small, f * csc_m[batch_id] * sizeof(float)));
			cudacall(cudaMemcpy(XT_small, &XT_h[(long) row_offset*f], f * csc_m[batch_id] * sizeof(float), cudaMemcpyHostToDevice));

			printf("cal train rmse in batch: %d/%d, nnz:%d, n(col): %d, \n",
					batch_id, GPU_COUNT, csc_nnz[batch_id], n);

			cudacall(cudaMalloc((void** ) &cscRowIndex_small,csc_nnz[batch_id] * sizeof(int)));
            cudacall(cudaMalloc((void** ) &cscColIndex_small, (n + 1) * sizeof(int)));
            cudacall(cudaMalloc((void** ) &cscVal_small, csc_nnz[batch_id] * sizeof(float)));
            cudacall(cudaMemcpy(cscRowIndex_small, cscRowIndexHostPtr[batch_id],
        			csc_nnz[batch_id] * sizeof(int), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(cscColIndex_small, cscColIndexHostPtr[batch_id],
        			(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(cscVal_small, cscValHostPtr[batch_id],
					csc_nnz[batch_id] * sizeof(float), cudaMemcpyHostToDevice));


			cudacall(cudaMalloc((void** ) &nan_train, sizeof(int)));
			cudacall( cudaMemset(nan_train, 0, sizeof(int)) );
			cudacall(cudaMalloc((void** ) &nan_test, sizeof(int)));
			cudacall( cudaMemset(nan_test, 0, sizeof(int)) );

			RMSE_CSC<<<n, 512>>>(cscVal_small, cscRowIndex_small,
					cscColIndex_small, thetaT[0], XT_small, errors_train, error_size, nan_train);
			cudaDeviceSynchronize();
			cudaCheckError();

			cudacall(cudaFree(cscRowIndex_small));
			cudacall(cudaFree(cscColIndex_small));
			cudacall(cudaFree(cscVal_small));

			printf("cal test rmse in batch: %d/%d, nnz_test:%d, n(col): %d, \n",
					batch_id, GPU_COUNT, csc_nnz_test[batch_id], n);
            cudacall(cudaMalloc((void** ) &cscRowIndex_small,csc_nnz_test[batch_id] * sizeof(int)));
            cudacall(cudaMalloc((void** ) &cscColIndex_small, (n + 1) * sizeof(int)));
            cudacall(cudaMalloc((void** ) &cscVal_small, csc_nnz_test[batch_id] * sizeof(float)));
            cudacall(cudaMemcpy(cscRowIndex_small, testCscRowIndexHostPtr[batch_id],
        			csc_nnz_test[batch_id] * sizeof(int), cudaMemcpyHostToDevice));
            cudacall(cudaMemcpy(cscColIndex_small, testCscColIndexHostPtr[batch_id],
        			(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cudacall(cudaMemcpy(cscVal_small, testCscValHostPtr[batch_id],
					csc_nnz_test[batch_id] * sizeof(float), cudaMemcpyHostToDevice));
			RMSE_CSC<<<n, 512>>>(cscVal_small, cscRowIndex_small,
					cscColIndex_small, thetaT[0], XT_small, errors_test, error_size, nan_test);
			cudaDeviceSynchronize();
			cudaCheckError();

			int* nan_train_host = (int*) malloc (sizeof(int));
			int* nan_test_host = (int*) malloc (sizeof(int));
			cudaMemcpy(nan_train_host, nan_train, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(nan_test_host, nan_test, sizeof(int), cudaMemcpyDeviceToHost);

			printf("train #nan: %d\n", *nan_train_host);
			printf("test #nan: %d\n", *nan_test_host);
			cudacall(cudaFree(nan_train));
			cudacall(cudaFree(nan_test));



			cudacall(cudaFree(cscRowIndex_small));
			cudacall(cudaFree(cscColIndex_small));
			cudacall(cudaFree(cscVal_small));

			cudacall(cudaFree(XT_small));

		}
		printf("summarize RMSE: \n");
		float* rmse_train = (float*) malloc (sizeof(float));
		cublascall( cublasSasum(handle[0], error_size, errors_train, 1, rmse_train) );
		cudaDeviceSynchronize();
		cudaCheckError();
		float* rmse_test = (float*) malloc (sizeof(float));
		cublascall( cublasSasum(handle[0], error_size, errors_test, 1, rmse_test) );
		cudaDeviceSynchronize();
		cudaCheckError();

		printf("@@@@@@@@@@@@@@@@@@@ Train RMSE in iter %d: %f\n", iter, sqrt((*rmse_train)/nnz));
		printf("@@@@@@@@@@@@@@@@@@@ Test RMSE in iter %d: %f\n", iter, sqrt((*rmse_test)/(NNZ_TEST - 12750)));

		cudacall(cudaFree(errors_train));
		cudacall(cudaFree(errors_test));
//*/
	}
	/*
	//save model to a file
	cudacall(cudaMemcpy(thetaTHost, thetaT[0], n * f * sizeof(float), cudaMemcpyDeviceToHost) );
	FILE * xfile = fopen("XT.data", "wb");
	FILE * thetafile = fopen("thetaT.data", "wb");
	fwrite(XT_h, sizeof(float), m*f, xfile);
	fwrite(thetaTHost, sizeof(float), n*f, thetafile);
	fclose(xfile);
	fclose(thetafile);
	*/

	cudacall(cudaFreeHost(XT_h));
	cudacall(cudaFreeHost(csrRowIndexHostPtr));
	cudacall(cudaFreeHost(csrColIndexHostPtr));
	cudacall(cudaFreeHost(csrValHostPtr));
	cudaFreeHost(thetaTHost);
	for(int gpu_id = 0; gpu_id < GPU_COUNT; gpu_id ++){
		cudacall(cudaFreeHost(cscValHostPtr[gpu_id]));
		cudacall(cudaFreeHost(cscRowIndexHostPtr[gpu_id]));
		cudacall(cudaFreeHost(cscColIndexHostPtr[gpu_id]));
		cudacall(cudaSetDevice(gpu_id));
		//cudacall(cudaDeviceReset());
	}
	printf("ALS Done.\n");
	return 0;
}
