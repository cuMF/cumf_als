/*
 * als_main.cu
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Test als.cu using netflix or yahoo data
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */
#include "als.h"
#include "host_utilities.h"
#include<stdio.h>

#define DEVICEID 0
#define ITERS 10

//netflix standard data
#define M 17770
#define N 480189
#define NNZ 99072112
#define NNZ_TEST 1408395
#define X_BATCH 1

//lambda: K40 and Maxwell: 0.055
//K80: needs 0.06

/*
//yahoo data
#define M 1000990
#define N 624961
#define NNZ 252800275
#define NNZ_TEST 4003960
//1.2 on K40, Maxwell
//need to be 2.0+ on K80
#define LAMBDA 1.1
#define THETA_BATCH 3
#define X_BATCH 6
*/

int main(int argc, char **argv) {

	if(argc!=4){
		printf("usage: give F, lambda and THETA_BATCH.\n");
		return 0;

	}
	else {
		printf("F = %s, lambda = %s, THETA_BATCH = %s \n", argv[1], argv[2], argv[3]);
	}
	int f = atoi(argv[1]);
	if(f%T10!=0){
		printf("F has to be a multiple of %d \n", T10);
		return 0;
	}

	cudaSetDevice(DEVICEID);

	int m = M;
	int n = N;
	long nnz = NNZ;
	long nnz_test = NNZ_TEST;
	float lambda = atof(argv[2]);
	int THETA_BATCH = atoi(argv[3]);

	int* csrRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrRowIndexHostPtr, (m + 1) * sizeof(csrRowIndexHostPtr[0])) );
	int* csrColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrColIndexHostPtr, nnz * sizeof(csrColIndexHostPtr[0])) );
	float* csrValHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrValHostPtr, nnz * sizeof(csrValHostPtr[0])) );
	float* cscValHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscValHostPtr, nnz * sizeof(cscValHostPtr[0])) );
	int* cscRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscRowIndexHostPtr, nnz * sizeof(cscRowIndexHostPtr[0])) );
	int* cscColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscColIndexHostPtr, (n+1) * sizeof(cscColIndexHostPtr[0])) );
	int* cooRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cooRowIndexHostPtr, nnz * sizeof(cooRowIndexHostPtr[0])) );

	//calculate X from thetaT first, need to initialize thetaT
	float* thetaTHost;
	cudacall(cudaMallocHost( (void** ) &thetaTHost, n * f * sizeof(thetaTHost[0])) );

	float* XTHost;
	cudacall(cudaMallocHost( (void** ) &XTHost, m * f * sizeof(XTHost[0])) );

	//initialize thetaT on host
	srand (time(0));
	for (int k = 0; k < n * f; k++)
		//netflix standard
		thetaTHost[k] = 0.05*((float) rand() / (RAND_MAX)) - 0.35;
		//yahoo
		//thetaTHost[k] = 3.0*((float) rand() / (RAND_MAX)) - 1.0f;
	printf("*******starting loading training and testing sets to host.\n");
	//testing set
	int* cooRowIndexTestHostPtr = (int *) malloc(
			nnz_test * sizeof(cooRowIndexTestHostPtr[0]));
	int* cooColIndexTestHostPtr = (int *) malloc(
			nnz_test * sizeof(cooColIndexTestHostPtr[0]));
	float* cooValHostTestPtr = (float *) malloc(nnz_test * sizeof(cooValHostTestPtr[0]));

	struct timeval tv0;
	gettimeofday(&tv0, NULL);

	loadCooSparseMatrixBin("./netflix/R_test_coo.data.bin", "./netflix/R_test_coo.row.bin","./netflix/R_test_coo.col.bin",
	//loadCooSparseMatrixBin("./yahoo/yahoo_R_test_coo.data.bin", "./yahoo/yahoo_R_test_coo.row.bin", "./yahoo/yahoo_R_test_coo.col.bin",
			cooValHostTestPtr, cooRowIndexTestHostPtr, cooColIndexTestHostPtr, nnz_test);

    loadCSRSparseMatrixBin("./netflix/R_train_csr.data.bin", "./netflix/R_train_csr.indptr.bin", "./netflix/R_train_csr.indices.bin",
    //loadCSRSparseMatrixBin("./yahoo/yahoo_R_train_csr.data.bin", "./yahoo/yahoo_R_train_csr.indptr.bin", "./yahoo/yahoo_R_train_csr.indices.bin",
    		csrValHostPtr, csrRowIndexHostPtr, csrColIndexHostPtr, m, nnz);

    loadCSCSparseMatrixBin("./netflix/R_train_csc.data.bin", "./netflix/R_train_csc.indices.bin", "./netflix/R_train_csc.indptr.bin",
    //loadCSCSparseMatrixBin("./yahoo/yahoo_R_train_csc.data.bin", "./yahoo/yahoo_R_train_csc.indices.bin", "./yahoo/yahoo_R_train_csc.indptr.bin",
   		cscValHostPtr, cscRowIndexHostPtr, cscColIndexHostPtr, n, nnz);

    loadCooSparseMatrixRowPtrBin("./netflix/R_train_coo.row.bin", cooRowIndexHostPtr, nnz);
    //loadCooSparseMatrixRowPtrBin("./yahoo/yahoo_R_train_coo.row.bin", cooRowIndexHostPtr, nnz);

	#ifdef DEBUG
    printf("\nloaded csr to host; print data, row and col array\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%.1f ", csrValHostPtr[i]);
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
	
	#endif
	double t0 = seconds();
	
	doALS(csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr,
			cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
			cooRowIndexHostPtr, thetaTHost, XTHost,
			cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValHostTestPtr,
			m, n, f, nnz, nnz_test, lambda,
			ITERS, X_BATCH, THETA_BATCH);
	printf("\ndoALS takes seconds: %.3f for F= %d\n", seconds() - t0, f);

	/*
	//write out the model	
	FILE * xfile = fopen("XT.data", "wb");
	FILE * thetafile = fopen("thetaT.data", "wb");
	fwrite(XTHost, sizeof(float), m*f, xfile);
	fwrite(thetaTHost, sizeof(float), n*f, thetafile);
	fclose(xfile);
	fclose(thetafile);
	*/

	cudaFreeHost(csrRowIndexHostPtr);
	cudaFreeHost(csrColIndexHostPtr);
	cudaFreeHost(csrValHostPtr);
	cudaFreeHost(cscValHostPtr);
	cudaFreeHost(cscRowIndexHostPtr);
	cudaFreeHost(cscColIndexHostPtr);
	cudaFreeHost(cooRowIndexHostPtr);
	cudaFreeHost(XTHost);
	cudaFreeHost(thetaTHost);
	cudacall(cudaDeviceReset());
	printf("\nALS Done.\n");
	return 0;
}

