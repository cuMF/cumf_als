#include <sys/time.h>
#include <iostream>
#include <cublas_v2.h>
#include "../als.h"
inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


template <typename T>
std::string to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}
void enableP2P(int numGPUs)
{
    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);
        for (int j=0; j<numGPUs; j++)
        {
            int access;
            cudaDeviceCanAccessPeer(&access,i,j);

            if (access)
            {
                cudaDeviceEnablePeerAccess(j,0);
                cudaCheckError();
            }
        }
    }
}

void loadCSCSparseMatrixInBatch(const std::string dataFile, const std::string rowFile, const std::string colFile, float * data, int* row, int* col, long long csc_nnz, int n) {
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
