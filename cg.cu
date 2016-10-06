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
#include "device_utilities.h"
#include "host_utilities.h"
#include <fstream>
#define SCAN_BATCH 24
#define CG_ERROR 1e-4
#undef DEBUG

//CG (iterative solve) kernel
//each block solves a A*x=b 
__global__ void updateXWithCGKernel(float * A, float * x, float * b, const float* yTy, const int batchSize, const int f, const float cgIter){
	extern __shared__ float smem[];
	float *sharedx = &smem[0];
	float *sharedp = &smem[f];
	float *sharedr = &smem[2*f];
	float *sharedap = &smem[3*f];
	float *rsold = &smem[4*f]; 
	float *alpha = &smem[4*f+1];
	float *rsnew = &smem[4*f+2];
	float *beta = &smem[4*f+3];
	
	//A+=yTy
	for(int i = 0; i < f; i++)
		A[blockIdx.x*f*f + i*f + threadIdx.x] += yTy[i*f + threadIdx.x];
	__syncthreads();

	//sharedx<--x
	sharedx[threadIdx.x] = x[blockIdx.x*blockDim.x + threadIdx.x];
	//sharedx[threadIdx.x] = 0;
	__syncthreads();
	//r=b-A*x;
	float temp = 0;
	for(int i = 0; i < f; i++)
		//this is math correct and coalesced because A is symmetric
		temp += A[blockIdx.x*f*f + f*i + threadIdx.x]*sharedx[i];
	sharedr[threadIdx.x] = b[blockIdx.x*blockDim.x + threadIdx.x] - temp;
	//p=r;
	sharedp[threadIdx.x] = sharedr[threadIdx.x];
	//rsold=r'*r;
	if(threadIdx.x == 0){
		rsold[0] = 0;
	}
	temp = sharedr[threadIdx.x]
			*sharedr[threadIdx.x];
	blockReduceSumWithAtomics(rsold, temp);	
    //temp = blockReduceSum(shared, temp);
	__syncthreads();
	#ifdef DEBUG
	if(threadIdx.x==0){
		printf("***rsold:\n");
		printf("rsold = %f \n", rsold[0]);
		printf("***shared memory content after 1st blockReduceSum:\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedp[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedr[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedap[i]);
		printf("\n");
	}
	#endif

	for(int iter = 0; iter < cgIter; iter++){
		//ap=A*p;
		//WARN: set temp to zero since the next operation is +=!
		temp = 0;
		for(int i = 0; i < f; i++)
			//this is math correct and coalesced because A is symmetric
			temp += A[blockIdx.x*f*f + f*i + threadIdx.x]*sharedp[i];
		sharedap[threadIdx.x] = temp;
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("----------CG iteration %d \n", iter);
			printf("***ap:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
			printf("***shared memory content before 2rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(threadIdx.x == 0){
			rsnew[0] = 0;
		}
		//no need to have sync before blockReduce
		//because there is a __syncthreads() in blockReduce
		//pAp=p'*Ap
		temp = sharedp[threadIdx.x]
				*sharedap[threadIdx.x];		
		//temp = blockReduceSum(shared, temp);
		blockReduceSumWithAtomics(rsnew, temp);
		//sync needed, to let all atomicAdd threads completes
		__syncthreads();
		if(threadIdx.x == 0){
			//pAp = temp;
			//alpha=rsold/(p'*Ap); use rsnew to store pAp
			alpha[0] = rsold[0]/rsnew[0];
			#ifdef DEBUG
			printf("***rsold:\n");
			printf("rsold = %f \n", rsold[0]);
			printf("***pAp:\n");
			printf("pAp = %f \n", rsnew[0]);
			printf("***alpha:\n");
			printf("alpha = %f \n", alpha[0]);
			#endif
			rsnew[0] = 0;
		}
		//needed, aplpha[0] to be used by all threads
		__syncthreads();
		//x=x+alpha*p;
		sharedx[threadIdx.x] = 
			sharedx[threadIdx.x] + alpha[0] * sharedp[threadIdx.x];
        //r=r-alpha*Ap;
		sharedr[threadIdx.x] = 
			sharedr[threadIdx.x] - alpha[0] * sharedap[threadIdx.x];
		//NOT needed?
		//__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content before 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		
		//rsnew=r'*r;
		/*
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		temp = blockReduceSum(shared, temp);
		__syncthreads();
		if(threadIdx.x == 0){
			rsnew[0] = temp;
		}
		*/
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		blockReduceSumWithAtomics(rsnew, temp);
		//WARN: has to have this sync!
		__syncthreads();

		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***rsnew:\n");
			printf("rsnew = %f \n", rsnew[0]);
			printf("***shared memory content after 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(rsnew[0]<CG_ERROR)
			break;
		//NOT needed?
		//__syncthreads();
		//beta
		if(threadIdx.x == 0){
			beta[0] = rsnew[0]/rsold[0];
			//rsold=rsnew;
			rsold[0] = rsnew[0];
		}
		//need sync since every thread needs beta[0]
		__syncthreads();
		//p=r+(rsnew/rsold)*p;
		sharedp[threadIdx.x] = 
			sharedr[threadIdx.x] + beta[0] * sharedp[threadIdx.x];
		//need sync as every thread needs sharedp at the beginning of for
		__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content after update p:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		__syncthreads();
		#endif
	}//end of CG iterations
	//x<--sharedx
	x[blockIdx.x*blockDim.x + threadIdx.x] = sharedx[threadIdx.x];
}

//CG (iterative solve) kernel
//each block solves a A*x=b and A in fp16
__global__ void updateXWithCGKernel3(half * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	extern __shared__ float smem[];
	float *sharedx = &smem[0];
	float *sharedp = &smem[f];
	float *sharedr = &smem[2*f];
	float *sharedap = &smem[3*f];
	float *rsold = &smem[4*f]; 
	float *alpha = &smem[4*f+1];
	float *rsnew = &smem[4*f+2];
	float *beta = &smem[4*f+3];

	//sharedx<--x
	sharedx[threadIdx.x] = x[blockIdx.x*blockDim.x + threadIdx.x];
	__syncthreads();
	//r=b-A*x;
	float temp = 0;
	for(int i = 0; i < f; i++)
		//this is math correct and coalesced because A is symmetric
		temp += __half2float(A[blockIdx.x*f*f + f*i + threadIdx.x])*sharedx[i];
	sharedr[threadIdx.x] = b[blockIdx.x*blockDim.x + threadIdx.x] - temp;
	//p=r;
	sharedp[threadIdx.x] = sharedr[threadIdx.x];
	//rsold=r'*r;
	if(threadIdx.x == 0){
		rsold[0] = 0;
	}
	temp = sharedr[threadIdx.x]
			*sharedr[threadIdx.x];
	blockReduceSumWithAtomics(rsold, temp);	
    //temp = blockReduceSum(shared, temp);
	__syncthreads();
	#ifdef DEBUG
	if(threadIdx.x==0){
		printf("***rsold:\n");
		printf("rsold = %f \n", rsold[0]);
		printf("***shared memory content after 1st blockReduceSum:\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedp[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedr[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedap[i]);
		printf("\n");
	}
	#endif

	for(int iter = 0; iter < cgIter; iter++){
		//ap=A*p;
		//WARN: set temp to zero since the next operation is +=!
		temp = 0;
		for(int i = 0; i < f; i++)
			//this is math correct and coalesced because A is symmetric
			temp += __half2float(A[blockIdx.x*f*f + f*i + threadIdx.x])*sharedp[i];
		sharedap[threadIdx.x] = temp;
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("----------CG iteration %d \n", iter);
			printf("***ap:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
			printf("***shared memory content before 2rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(threadIdx.x == 0){
			rsnew[0] = 0;
		}
		//no need to have sync before blockReduce
		//because there is a __syncthreads() in blockReduce
		//pAp=p'*Ap
		temp = sharedp[threadIdx.x]
				*sharedap[threadIdx.x];		
		//temp = blockReduceSum(shared, temp);
		blockReduceSumWithAtomics(rsnew, temp);
		//sync needed, to let all atomicAdd threads completes
		__syncthreads();
		if(threadIdx.x == 0){
			//pAp = temp;
			//alpha=rsold/(p'*Ap); use rsnew to store pAp
			alpha[0] = rsold[0]/rsnew[0];
			#ifdef DEBUG
			printf("***rsold:\n");
			printf("rsold = %f \n", rsold[0]);
			printf("***pAp:\n");
			printf("pAp = %f \n", rsnew[0]);
			printf("***alpha:\n");
			printf("alpha = %f \n", alpha[0]);
			#endif
			rsnew[0] = 0;
		}
		//needed, aplpha[0] to be used by all threads
		__syncthreads();
		//x=x+alpha*p;
		sharedx[threadIdx.x] = 
			sharedx[threadIdx.x] + alpha[0] * sharedp[threadIdx.x];
        //r=r-alpha*Ap;
		sharedr[threadIdx.x] = 
			sharedr[threadIdx.x] - alpha[0] * sharedap[threadIdx.x];
		//NOT needed?
		//__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content before 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		
		//rsnew=r'*r;
		/*
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		temp = blockReduceSum(shared, temp);
		__syncthreads();
		if(threadIdx.x == 0){
			rsnew[0] = temp;
		}
		*/
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		blockReduceSumWithAtomics(rsnew, temp);
		//WARN: has to have this sync!
		__syncthreads();

		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***rsnew:\n");
			printf("rsnew = %f \n", rsnew[0]);
			printf("***shared memory content after 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(rsnew[0]<CG_ERROR)
			break;
		//NOT needed?
		//__syncthreads();
		//beta
		if(threadIdx.x == 0){
			beta[0] = rsnew[0]/rsold[0];
			//rsold=rsnew;
			rsold[0] = rsnew[0];
		}
		//need sync since every thread needs beta[0]
		__syncthreads();
		//p=r+(rsnew/rsold)*p;
		sharedp[threadIdx.x] = 
			sharedr[threadIdx.x] + beta[0] * sharedp[threadIdx.x];
		//need sync as every thread needs sharedp at the beginning of for
		__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content after update p:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		__syncthreads();
		#endif
	}//end of CG iterations
	//x<--sharedx
	x[blockIdx.x*blockDim.x + threadIdx.x] = sharedx[threadIdx.x];
}

//blockDim.x=64 or 96 (two or three WARPs) instead of 100 -- WARP shuffle seems requiring this
__global__ void updateXWithCGKernel2(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	extern __shared__ float smem[];
	float *sharedx = &smem[0];
	float *sharedp = &smem[f];
	float *sharedr = &smem[2*f];
	float *sharedap = &smem[3*f];
	float *rsold = &smem[4*f]; 
	float *alpha = &smem[4*f+1];
	float *rsnew = &smem[4*f+2];
	float *beta = &smem[4*f+3];

	//sharedx<--x
	for(int k = threadIdx.x; k < f; k += blockDim.x)
		sharedx[k] = x[blockIdx.x*f + k];
	__syncthreads();
	//r=b-A*x;
	float temp = 0;
	for(int k = threadIdx.x; k < f; k += blockDim.x){
		temp = 0;
		for(int i = 0; i < f; i++)
			temp += A[blockIdx.x*f*f + f*i + k]*sharedx[i];
		sharedr[k] = b[blockIdx.x*f + k] - temp;
		//p=r;
		sharedp[k] = sharedr[k];
	}
	//rsold=r'*r;
	if(threadIdx.x == 0){
		rsold[0] = 0;
	}
	temp = 0;
	for(int k = threadIdx.x; k < f; k += blockDim.x){
		temp += sharedr[k]*sharedr[k];
	}
	blockReduceSumWithAtomics(rsold, temp);	
    //temp = blockReduceSum(shared, temp);
	__syncthreads();
	#ifdef DEBUG
	if(threadIdx.x==0){
		printf("***rsold:\n");
		printf("rsold = %f \n", rsold[0]);
		printf("***shared memory content after 1st blockReduceSum:\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedp[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedr[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedap[i]);
		printf("\n");
	}
	#endif

	for(int iter = 0; iter < cgIter; iter++){
		//ap=A*p;
		//WARN: set temp to zero since the next operation is +=!
		for(int k = threadIdx.x; k < f; k += blockDim.x){
			temp = 0;
			for(int i = 0; i < f; i++)
				temp += A[blockIdx.x*f*f + f*i + k]*sharedp[i];
			sharedap[k] = temp;
		}
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("----------CG iteration %d \n", iter);
			printf("***ap:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
			printf("***shared memory content before 2rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(threadIdx.x == 0){
			rsnew[0] = 0;
		}
		//no need to have sync before blockReduce
		//because there is a __syncthreads() in blockReduce
		//pAp=p'*Ap
		temp = 0;
		for(int k = threadIdx.x; k < f; k += blockDim.x)
			temp += sharedp[k]*sharedap[k];		
		//temp = blockReduceSum(shared, temp);
		blockReduceSumWithAtomics(rsnew, temp);
		//sync needed, to let all atomicAdd threads completes
		__syncthreads();
		if(threadIdx.x == 0){
			//pAp = temp;
			//alpha=rsold/(p'*Ap); use rsnew to store pAp
			alpha[0] = rsold[0]/rsnew[0];
			#ifdef DEBUG
			printf("***rsold:\n");
			printf("rsold = %f \n", rsold[0]);
			printf("***pAp:\n");
			printf("pAp = %f \n", rsnew[0]);
			printf("***alpha:\n");
			printf("alpha = %f \n", alpha[0]);
			#endif
			rsnew[0] = 0;
		}
		//needed, aplpha[0] to be used by all threads
		__syncthreads();
		for(int k = threadIdx.x; k < f; k += blockDim.x){
			//x=x+alpha*p;
			sharedx[k] = 
				sharedx[k] + alpha[0] * sharedp[k];
			//r=r-alpha*Ap;
			sharedr[k] = 
				sharedr[k] - alpha[0] * sharedap[k];
		}
		//NOT needed?
		//__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content before 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		
		//rsnew=r'*r;
		/*
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		temp = blockReduceSum(shared, temp);
		__syncthreads();
		if(threadIdx.x == 0){
			rsnew[0] = temp;
		}
		*/
		temp = 0;
		for(int k = threadIdx.x; k < f; k += blockDim.x)
			temp += sharedr[k]*sharedr[k];
		blockReduceSumWithAtomics(rsnew, temp);
		//WARN: has to have this sync!
		__syncthreads();

		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***rsnew:\n");
			printf("rsnew = %f \n", rsnew[0]);
			printf("***shared memory content after 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(rsnew[0]<CG_ERROR)
			break;
		//NOT needed?
		//__syncthreads();
		//beta
		if(threadIdx.x == 0){
			beta[0] = rsnew[0]/rsold[0];
			//rsold=rsnew;
			rsold[0] = rsnew[0];
		}
		//need sync since every thread needs beta[0]
		__syncthreads();
		for(int k = threadIdx.x; k < f; k += blockDim.x)
			//p=r+(rsnew/rsold)*p;
			sharedp[k] = 
				sharedr[k] + beta[0] * sharedp[k];
		//need sync as every thread needs sharedp at the beginning of for
		__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content after update p:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		__syncthreads();
		#endif
	}//end of CG iterations
	for(int k = threadIdx.x; k < f; k += blockDim.x)
		//x<--sharedx
		x[blockIdx.x*f + k] = sharedx[k];
}

void updateXWithCGHost_tt_fp16(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	updateXWithCGKernel3<<<batchSize, f, (4*f+4)*sizeof(float)>>>
		((half*)A, x, b, batchSize, f, cgIter);
	cudaDeviceSynchronize();
	cudaCheckError();
	
	#ifdef DEBUG
	
	printf("***A[0]:\n");
	float *h_A = new float[f * f];
	float *A_fp32;
	cudacall(cudaMalloc((void** ) &A_fp32, f * f * sizeof(A_fp32[0])));
	fp16Array2fp32Array<<<(f*f-1)/1024 + 1, 1024>>>(A_fp32, (half*)A, f*f);
	cudaDeviceSynchronize();
	cudaCheckError();
	cudacall(cudaMemcpy(h_A, A_fp32, f * f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f*f; i++)
		printf("%f ", h_A[i]);
	printf("\n");
	delete [] h_A;
	cudacall(cudaFree(A_fp32));
	
	printf("***x[0]:\n");
	float *h_x = new float[f];
	cudacall(cudaMemcpy(h_x, x, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_x[i]);
	printf("\n");
	delete [] h_x;
	/*
	printf("***b[0]:\n");
	float *h_b = new float[f];
	cudacall(cudaMemcpy(h_b, b, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_b[i]);
	printf("\n");
	delete [] h_b;
	*/
	#endif
}

void updateXWithCGHost(float * A, float * x, float * b, const float* yTy, const int batchSize, const int f, const float cgIter){
	updateXWithCGKernel<<<batchSize, f, (4*f+4)*sizeof(float)>>>
	//updateXWithCGKernel2<<<batchSize, 96, (4*f+4)*sizeof(float)>>>
		(A, x, b, yTy, batchSize, f, cgIter);
	cudaDeviceSynchronize();
	cudaCheckError();
	
	#ifdef DEBUG
	
	printf("***A[0]:\n");
	float *h_A = new float[f * f];
	float *A_fp32;
	cudacall(cudaMalloc((void** ) &A_fp32, f * f * sizeof(A_fp32[0])));
	fp16Array2fp32Array<<<(f*f-1)/1024 + 1, 1024>>>(A_fp32, (half*)A, f*f);
	cudaDeviceSynchronize();
	cudaCheckError();
	cudacall(cudaMemcpy(h_A, A_fp32, f * f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f*f; i++)
		printf("%f ", h_A[i]);
	printf("\n");
	delete [] h_A;
	cudacall(cudaFree(A_fp32));
	
	printf("***x[0]:\n");
	float *h_x = new float[f];
	cudacall(cudaMemcpy(h_x, x, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_x[i]);
	printf("\n");
	delete [] h_x;
	/*
	printf("***b[0]:\n");
	float *h_b = new float[f];
	cudacall(cudaMemcpy(h_b, b, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_b[i]);
	printf("\n");
	delete [] h_b;
	*/
	#endif
}		
