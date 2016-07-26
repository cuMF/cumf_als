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
 * cg.cu
 *  Created on: July 22, 2016
 *  Author: Wei Tan (wtan@us.ibm.com)
 *  CUDA kernels related to batch CG solver used in ALS
 *	CG solver: https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */

#include "als.h"
#include "host_utilities.h"
#include <fstream>
#include <assert.h>
#include <cuda_fp16.h> 

#define WARP_SIZE 32

//WARP shuffling code adopted from here:
//https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {

  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);     // Each warp performs partial reduction
  #ifdef DEBUG
  //printf("warp id %d, lane %d, val: %f \n", wid, lane, val);
  #endif
  //aggregate the zero thread in every wrap 
  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x <= blockDim.x / WARP_SIZE) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
  #ifdef DEBUG
  //printf("RETURN block id %d, wid %d, lane %d, blockReduceSum: %f \n", blockIdx.x, wid, lane, val);
  #endif
  return val;
}

//result[i] = x[i] dot y[i]
//one block per (x[i], y[i])
__global__ void sdot_batched(const float * x, const float * y, const int elementSize, 
	const int batchSize, float * result) {
	float tmp_product = x[blockIdx.x*blockDim.x + threadIdx.x]
							*y[blockIdx.x*blockDim.x + threadIdx.x];
	#ifdef DEBUG
	//printf("%f\n", tmp_product);
	#endif
	tmp_product = blockReduceSum(tmp_product);
	#ifdef DEBUG
	//printf("block %d, thread %d, blockSum: %f\n", blockIdx.x, threadIdx.x, tmp_product);
	#endif
	if(threadIdx.x == 0){
		result[blockIdx.x] = tmp_product;
		//printf("block %d, thread %d, result: %f\n", blockIdx.x, threadIdx.x, tmp_product);
	}
}

//y[i] = alpha* A[i]*x[i]
//one block per (A[i],x[i])
//blockDim.x==m==n; A = A^T
__global__ void symmetric_sgemv_batched(float alpha, int m, const float * A, const float * x, float * y,
	const int batchSize) {
	extern __shared__ float sharedx[];
	sharedx[threadIdx.x] = x[blockIdx.x*blockDim.x + threadIdx.x];
	y[threadIdx.x] = 0;
	__syncthreads();	
	float temp = 0;
	for(int i = 0; i < m; i++)
		//this is math correct and coalesced because A is symmetric
		temp += A[blockIdx.x*m*m + m*i + threadIdx.x]*x[blockIdx.x*m + i];
	y[blockIdx.x*blockDim.x + threadIdx.x] = alpha*temp;
	#ifdef DEBUG
	//printf("block %d, y[%d]=%f \n", blockIdx.x, threadIdx.x, y[blockIdx.x*blockDim.x + threadIdx.x]);
	#endif
}

//y=y/x, element wise
__global__ void elementWiseDivide(float * result, const float * y, float* x, const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		float temp =  y[i]/x[i];
		//TODO: this is a temporary solution
		if(isnan(temp)||!isfinite(temp)) temp = 0;
		result[i] = temp;
	}
}

//y[i] = alpha[i]*x[i] + y[i]; i = 0,1,...k-1
//each y[i] is of size f*1
//MUST launch with <<<k,f>>>
__global__ void saxpyBatched(
                           const float           *alpha,
                           const float           *x,
                           float                 *y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	y[i] = alpha[blockIdx.x]*x[i] + y[i];					   
}

__global__ void saxpyBatched2(
                           const float           *alpha,
                           const float           *x,
                           float                 *y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	y[i] = -alpha[blockIdx.x]*x[i] + y[i];					   
}



//fused kernel
//each block solves a A*x=b 
__global__ void updateXWithCGKernel(float * A, float * x, float * b, const int batchSize, const int f,
								float *r, float *p, float *rsold, 
								float *ap, float *pAp, float *alpha, float *rsnew, float *beta,
								const float cgIter){
	//r=b-A*x;
	extern __shared__ float sharedx[];
	sharedx[threadIdx.x] = x[blockIdx.x*blockDim.x + threadIdx.x];
	__syncthreads();
	float temp = 0;
	for(int i = 0; i < f; i++)
		//this is math correct and coalesced because A is symmetric
		temp += A[blockIdx.x*f*f + f*i + threadIdx.x]*sharedx[i];
	r[blockIdx.x*blockDim.x + threadIdx.x] = b[blockIdx.x*blockDim.x + threadIdx.x] - temp;
	//p=r;
	p[blockIdx.x*blockDim.x + threadIdx.x] = r[blockIdx.x*blockDim.x + threadIdx.x];
	__syncthreads();
	//rsold=r'*r;
	temp = r[blockIdx.x*blockDim.x + threadIdx.x]
			*r[blockIdx.x*blockDim.x + threadIdx.x];
    temp = blockReduceSum(temp);
	if(threadIdx.x == 0){
		rsold[blockIdx.x] = temp;
		//printf("block %d, thread %d, result: %f\n", blockIdx.x, threadIdx.x, tmp_product);
	}
	__syncthreads();
	#ifdef DEBUG
	if(threadIdx.x==0){
		printf("***rsold:\n");
		printf("rsold[%d] = %f \n", blockIdx.x, rsold[blockIdx.x]);
	}
	#endif

	for(int iter = 0; iter < cgIter; iter++){
		//ap=A*p;
		//sharedx <-- p
		sharedx[threadIdx.x] = p[blockIdx.x*blockDim.x + threadIdx.x];
		__syncthreads();
		//WARN: set temp to zero since the next operation is +=!
		temp = 0;
		for(int i = 0; i < f; i++)
			//this is math correct and coalesced because A is symmetric
			temp += A[blockIdx.x*f*f + f*i + threadIdx.x]*sharedx[i];
		ap[blockIdx.x*blockDim.x + threadIdx.x] = temp;
		__syncthreads();
		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***ap:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", ap[blockIdx.x*blockDim.x + i]);
			printf("\n");
		}
		#endif
		//pAp=p'*Ap
		temp = sharedx[threadIdx.x]
				*ap[blockIdx.x*blockDim.x + threadIdx.x];
		temp = blockReduceSum(temp);
		if(threadIdx.x == 0){
			pAp[blockIdx.x] = temp;
			//alpha=rsold/(p'*Ap);
			alpha[blockIdx.x] = rsold[blockIdx.x]/temp;
			//printf("block %d, thread %d, result: %f\n", blockIdx.x, threadIdx.x, tmp_product);
		}
		__syncthreads();
		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***pAp:\n");
			printf("pAp[%d] = %f \n", blockIdx.x, pAp[blockIdx.x]);
			printf("***alpha:\n");
			printf("alpha[%d] = %f \n", blockIdx.x, alpha[blockIdx.x]);
		}
		#endif
		//x=x+alpha*p;
		x[blockIdx.x*blockDim.x + threadIdx.x] = 
			x[blockIdx.x*blockDim.x + threadIdx.x] + alpha[blockIdx.x] * sharedx[threadIdx.x];
        //r=r-alpha*Ap;
		r[blockIdx.x*blockDim.x + threadIdx.x] = 
			r[blockIdx.x*blockDim.x + threadIdx.x] - alpha[blockIdx.x] * ap[blockIdx.x*blockDim.x + threadIdx.x];
		__syncthreads();
		//rsnew=r'*r;
		temp = r[blockIdx.x*blockDim.x + threadIdx.x]
					*r[blockIdx.x*blockDim.x + threadIdx.x];
		temp = blockReduceSum(temp);
		if(threadIdx.x == 0){
			rsnew[blockIdx.x] = temp;
			//printf("block %d, thread %d, result: %f\n", blockIdx.x, threadIdx.x, tmp_product);
		}
		__syncthreads();
		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***rsnew:\n");
			printf("rsnew[%d] = %f \n", blockIdx.x, rsnew[blockIdx.x]);
		}
		#endif
		if(rsnew[blockIdx.x]<1e-4)
			break;
		//beta
		if(threadIdx.x == 0){
			beta[blockIdx.x] = rsnew[blockIdx.x]/rsold[blockIdx.x];
			//rsold=rsnew;
			rsold[blockIdx.x] = rsnew[blockIdx.x];
		}
		//p=r+(rsnew/rsold)*p;
		p[blockIdx.x*blockDim.x + threadIdx.x] = 
			r[blockIdx.x*blockDim.x + threadIdx.x] + beta[blockIdx.x] * sharedx[threadIdx.x];
		//__syncthreads();
	}
}


void updateXWithCGHost(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	float *r, *p, *rsold;
	cudacall(cudaMalloc((void** ) &r, batchSize * f* sizeof(float)));
	cudacall(cudaMalloc((void** ) &p, batchSize * f* sizeof(float)));
	cudacall(cudaMalloc((void** ) &rsold, batchSize * sizeof(float)));
	float *ap;
	cudacall(cudaMalloc((void** ) &ap, batchSize * f* sizeof(float)));
	float *pAp;
	cudacall(cudaMalloc((void** ) &pAp, batchSize * sizeof(float)));
	float * alpha;
	cudacall(cudaMalloc((void** ) &alpha, batchSize * sizeof(float)));
	float * rsnew;
	cudacall(cudaMalloc((void** ) &rsnew, batchSize * sizeof(float)));
	float * beta;
	cudacall(cudaMalloc((void** ) &beta, batchSize * sizeof(float)));
	updateXWithCGKernel<<<batchSize, f, f*sizeof(float)>>>
		(A, x, b, batchSize, f,
		r, p, rsold, 
		ap, pAp, alpha, rsnew, beta,
		cgIter);
	cudaDeviceSynchronize();
	cudaCheckError();
	#ifdef DEBUG
	printf("***A[0]:\n");
	float *h_A = new float[f * f];
	cudacall(cudaMemcpy(h_A, A, f * f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f*f; i++)
		printf("%f ", h_A[i]);
	printf("\n");
	delete [] h_A;
	printf("***x[0]:\n");
	float *h_x = new float[f];
	cudacall(cudaMemcpy(h_x, x, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_x[i]);
	printf("\n");
	delete [] h_x;
	printf("***b[0]:\n");
	float *h_b = new float[f];
	cudacall(cudaMemcpy(h_b, b, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_b[i]);
	printf("\n");
	delete [] h_b;
	#endif

	cudacall(cudaFree(r));
	cudacall(cudaFree(p));
	cudacall(cudaFree(rsold));
	cudacall(cudaFree(ap));
	cudacall(cudaFree(pAp));
	cudacall(cudaFree(alpha));
	cudacall(cudaFree(rsnew));
	cudacall(cudaFree(beta));
}
//tt[i]*XT[i]=ythetaT[i]
//A*x=b
int updateXWithCG(const int batchSize, const int batchOffset, float * ythetaT, float * tt, float * XT,
		cublasHandle_t handle, const int m, const int n, const int f, const int nnz){
	#ifdef DEBUG
	double t0 = seconds();
	printf("*******Solve with CG.\n");
	#endif
	float *r, *p, *rsold;
	cudacall(cudaMalloc((void** ) &r, batchSize * f* sizeof(float)));
	cudacall(cudaMalloc((void** ) &p, batchSize * f* sizeof(float)));
	cudacall(cudaMalloc((void** ) &rsold, batchSize * sizeof(float)));
	//TODO x_0=0, or simply use the value from last iteration?
	//cudacall( cudaMemset(&XT[batchOffset*f], 0, f*batchSize*sizeof(float)) );
	cudaDeviceSynchronize();
	cudaCheckError();
	//r=b-A*x;
	//tt[i] * XT[i]
	const float h_alpha = 1.0f;
	symmetric_sgemv_batched<<<batchSize, f, f*sizeof(float)>>>
		(-h_alpha, f, tt, &XT[batchOffset*f], r, batchSize);
	cudaDeviceSynchronize();
	cudaCheckError();
	
	#ifdef DEBUG
	printf("***A[0]:\n");
	float *h_A = new float[f * f];
	cudacall(cudaMemcpy(h_A, tt,f * f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f*f; i++)
		printf("%f ", h_A[i]);
	printf("\n");
	delete [] h_A;
	printf("***x[0]:\n");
	float *h_x = new float[f];
	cudacall(cudaMemcpy(h_x, &XT[batchOffset*f], f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_x[i]);
	printf("\n");
	delete [] h_x;
	printf("***b[0]:\n");
	float *h_b = new float[f];
	cudacall(cudaMemcpy(h_b, &ythetaT[batchOffset*f], f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_b[i]);
	printf("\n");
	delete [] h_b;
	#endif
	
	#ifdef DEBUG
	printf("***r = -Ax:\n");
	float *h_r = new float[f * batchSize];
	cudacall(cudaMemcpy(h_r, r,f * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < batchSize*f; i++)
		printf("%f ", h_r[i]);
	printf("\n");
	#endif
	cublascall(cublasSaxpy(handle, batchSize*f, &h_alpha, &ythetaT[batchOffset * f], 1, r , 1));
	cudaDeviceSynchronize();
	cudaCheckError();
	#ifdef DEBUG
	printf("***r = b-Ax:\n");
	cudacall(cudaMemcpy(h_r, r,f * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < batchSize*f; i++)
		printf("%f ", h_r[i]);
	printf("\n");
	delete [] h_r;
	#endif
	//p=r;
	cudacall( cudaMemcpy(p, r, batchSize * f * sizeof(float), cudaMemcpyDeviceToDevice) );
	//rsold=r'*r;
	sdot_batched<<<batchSize, f>>>(r, r, f, batchSize, rsold);
	cudaDeviceSynchronize();
	cudaCheckError();
	#ifdef DEBUG
	printf("***rsold:\n");
	float *h_rsold = new float[batchSize];
	cudacall(cudaMemcpy(h_rsold, rsold, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < batchSize; i++)
		printf("%f ", h_rsold[i]);
	printf("\n");
	delete [] h_rsold;
	#endif

	float *Ap;
	cudacall(cudaMalloc((void** ) &Ap, batchSize * f* sizeof(float)));
	float *pAp;
	cudacall(cudaMalloc((void** ) &pAp, batchSize * sizeof(float)));
	float * alpha;
	cudacall(cudaMalloc((void** ) &alpha, batchSize * sizeof(float)));
	float * rsnew;
	cudacall(cudaMalloc((void** ) &rsnew, batchSize * sizeof(float)));
	float * rsnewDivideRsold;
	cudacall(cudaMalloc((void** ) &rsnewDivideRsold, batchSize * sizeof(float)));
	float * temp_r;
	cudacall(cudaMalloc((void** ) &temp_r, batchSize * f * sizeof(float)));

	for(int cgIter = 0; cgIter < 15; cgIter++){
		#ifdef DEBUG
		//printf("*******Start CG iterations %d.\n", cgIter);
		#endif
		//ap=A*p;
		symmetric_sgemv_batched<<<batchSize, f, f*sizeof(float)>>>
			(h_alpha, f, tt, p, Ap, batchSize);
		cudaDeviceSynchronize();
		cudaCheckError();
		//pAp=p'*Ap
		sdot_batched<<<batchSize, f>>>(p, Ap, f, batchSize, pAp);
		cudaDeviceSynchronize();
		cudaCheckError();
		//alpha=rsold/(p'*Ap);
		elementWiseDivide<<<(batchSize-1)/256 + 1, 256>>>(alpha, rsold, pAp, batchSize);
		cudaDeviceSynchronize();
		cudaCheckError();
		#ifdef DEBUG
		printf("***alpha_0:\n");
		float *h_alpha0 = new float[batchSize];
		cudacall(cudaMemcpy(h_alpha0, alpha,batchSize * sizeof(float), cudaMemcpyDeviceToHost));
		for(int i = 0; i < batchSize; i++)
		printf("%f ", h_alpha0[i]);
		printf("\n");
		delete [] h_alpha0;
		#endif

		//batched saxpy to update x (XT) and r
		//x=x+alpha*p;
		saxpyBatched<<<batchSize, f>>>(alpha, p, &XT[batchOffset*f]);
		cudaDeviceSynchronize();
		cudaCheckError();
		#ifdef DEBUG
		printf("***x=x+alpha*p:\n");
		float *h_x = new float[batchSize*f];
		cudacall(cudaMemcpy(h_x, &XT[batchOffset*f],batchSize * f* sizeof(float), cudaMemcpyDeviceToHost));
		for(int i = 0; i < batchSize*f; i++)
		printf("%f ", h_x[i]);
		printf("\n");
		delete [] h_x;
		#endif
        //r=r-alpha*Ap;
		saxpyBatched2<<<batchSize, f>>>(alpha, Ap, r);
		cudaDeviceSynchronize();
		cudaCheckError();
		#ifdef DEBUG
		printf("***r=r-alpha*Ap:\n");
		float *h_r = new float[f * batchSize];
		cudacall(cudaMemcpy(h_r, r,batchSize * f* sizeof(float), cudaMemcpyDeviceToHost));
		for(int i = 0; i < batchSize*f; i++)
		printf("%f ", h_r[i]);
		printf("\n");
		#endif
		//printf("*******rsnew \n");

		//rsnew=r'*r;
		//sdot_batched<<<batchSize, f>>>(r, r, f, batchSize, rsold);
		sdot_batched<<<batchSize, f>>>(r, r, f, batchSize, rsnew);
		cudaDeviceSynchronize();
		cudaCheckError();
		#ifdef DEBUG
		float *h_rsnew = new float[batchSize];
		cudacall(cudaMemcpy(h_rsnew, rsnew,batchSize * sizeof(float), cudaMemcpyDeviceToHost));
		printf("***rsnew: \n");
		for(int i = 0; i < batchSize; i++)
		printf("%.38f\t%e\t", h_rsnew[i], h_rsnew[i]);
		printf("\n");
		delete [] h_rsnew;
		#endif

		//TODO if sqrt(rsnew)<1e-10 break that particular batch;
		//how to skip: skip list???
		//printf("***beta: \n");

		//p=r+(rsnew/rsold)*p;
		//TODO when rsold (the rsnew in the last iteration) is too small, beta = rsnew/rsold might be NaN
		elementWiseDivide<<<(batchSize-1)/256 + 1, 256>>>(rsnewDivideRsold, rsnew, rsold, batchSize);
		cudaDeviceSynchronize();
		cudaCheckError();
		#ifdef DEBUG
		float *h_beta = new float[batchSize];
		cudacall(cudaMemcpy(h_beta, rsnewDivideRsold,batchSize * sizeof(float), cudaMemcpyDeviceToHost));
		printf("***beta: \n");
		for(int i = 0; i < batchSize; i++)
		printf("%.9f ", h_beta[i]);
		printf("\n");
		delete [] h_beta;
		#endif
		//printf("*******update p \n");
		//temp_r = r
		cudacall( cudaMemcpy(temp_r, r, batchSize * f * sizeof(float), cudaMemcpyDeviceToDevice) );
		//temp_r = temp_r + (rsnew/rsold)*p
		saxpyBatched<<<batchSize, f>>>(rsnewDivideRsold, p, temp_r);
		cudaDeviceSynchronize();
		cudaCheckError();
		//p = temp_r		
		cudacall( cudaMemcpy(p, temp_r, batchSize * f * sizeof(float), cudaMemcpyDeviceToDevice) );
        //rsold=rsnew;
		cudacall( cudaMemcpy(rsold, rsnew, batchSize * sizeof(float), cudaMemcpyDeviceToDevice) );
	}
	printf("*******Complte CG iterations.\n");
	cudacall(cudaFree(r));
	cudacall(cudaFree(p));
	cudacall(cudaFree(rsold));
	cudacall(cudaFree(Ap));
	cudacall(cudaFree(pAp));
	cudacall(cudaFree(alpha));
	cudacall(cudaFree(rsnew));
	cudacall(cudaFree(rsnewDivideRsold));
	cudacall(cudaFree(temp_r));

	#ifdef DEBUG
	printf("\t %f seconds. \n", seconds() - t0);
	#endif
	return 0;
}