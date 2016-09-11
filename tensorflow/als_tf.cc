#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
/* This is NOT a TF GPU op, instead it is a CPU op invoking GPUs
*  In future we should wrap individual cuMF kernels as TF ops, 
*  and use TF to write the glue code which is currently in c
*/
REGISTER_OP("DoAls")
    .Input("csrrow: int32")
	.Input("csrcol: int32")
	.Input("csrval: float")
	.Input("cscrow: int32")
	.Input("csccol: int32")
	.Input("cscval: float")
	.Input("coorow: int32")
	.Input("coorowtest: int32")
	.Input("coocoltest: int32")
	.Input("coovaltest: float")
	.Input("m_t:  int32")
	.Input("n_t:  int32")
	.Input("f_t:  int32")
	.Input("nnz_t:  int64")
	.Input("nnz_test_t:  int64")
	.Input("lambda_t:  float")
	.Input("iters_t:  int32")
	.Input("xbatch_t:  int32")
	.Input("thetabatch_t:  int32")
	.Input("deviceid_t:  int32")
    .Output("thetat: float")
	.Output("xt: float")
	.Output("rmse: float");
	using namespace tensorflow;

float doALS(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const float* csrValHostPtr,
		const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const float* cscValHostPtr,
		const int* cooRowIndexHostPtr, float* thetaTHost, float * XTHost,
		const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const float * cooValHostTestPtr,
		const int m, const int n, const int f, const long nnz, const long nnz_test, const float lambda,
		const int ITERS, const int X_BATCH, const int THETA_BATCH, const int DEVICEID);
	
class DoAlsOp : public OpKernel {
 public:
  explicit DoAlsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& csrrow_tensor = context->input(0);
	const Tensor& csrcol_tensor = context->input(1);
	const Tensor& csrval_tensor = context->input(2);
	const Tensor& cscrow_tensor = context->input(3);
	const Tensor& csccol_tensor = context->input(4);
	const Tensor& cscval_tensor = context->input(5);
	const Tensor& coorow_tensor = context->input(6);
    const Tensor& coorowtest_tensor = context->input(7);
	const Tensor& coocoltest_tensor = context->input(8);
	const Tensor& coovaltest_tensor = context->input(9);

	const Tensor& m_tensor = context->input(10);
	const Tensor& n_tensor = context->input(11);
	const Tensor& f_tensor = context->input(12);
	const Tensor& nnz_tensor = context->input(13);
	const Tensor& nnztest_tensor = context->input(14);
	
	const Tensor& lambda_tensor = context->input(15);
	const Tensor& iters_tensor = context->input(16);
	const Tensor& xbatch_tensor = context->input(17);
	const Tensor& thetabatch_tensor = context->input(18);
	const Tensor& deviceid_tensor = context->input(19);

    auto csrrow = csrrow_tensor.flat<int32>();
    auto csrcol = csrcol_tensor.flat<int32>();
    auto csrval = csrval_tensor.flat<float>();
	auto cscrow = cscrow_tensor.flat<int32>();
    auto csccol = csccol_tensor.flat<int32>();
    auto cscval = cscval_tensor.flat<float>();
    auto coorow = coorow_tensor.flat<int32>();
    auto coorowtest = coorowtest_tensor.flat<int32>();
    auto coocoltest = coocoltest_tensor.flat<int32>();
    auto coovaltest = coovaltest_tensor.flat<float>();

    auto m_t = m_tensor.flat<int32>();
    auto n_t = n_tensor.flat<int32>();
	auto f_t = f_tensor.flat<int32>();
	auto nnz_t = nnz_tensor.flat<int64>();
    auto nnztest_t = nnztest_tensor.flat<int64>();
    auto lambda_t = lambda_tensor.flat<float>();
	auto iters_t = iters_tensor.flat<int32>();
	auto xbatch_t = xbatch_tensor.flat<int32>();
	auto thetabatch_t = thetabatch_tensor.flat<int32>();
	auto deviceid_t = deviceid_tensor.flat<int32>();


    //const int N = csrrow.size();
	const int N = 2;
	const long nnz = nnz_t(0);
	const long nnztest = nnztest_t(0);
	const int m = m_t(0);
	const int n = n_t(0);
	const int f = f_t(0);
	const int iters = iters_t(0);
	const float lambda = lambda_t(0);
	const int xbatch = xbatch_t(0);
	const int thetabatch = thetabatch_t(0);
	const int deviceid = deviceid_t(0);

	// Create an output tensor
    Tensor* thetat_tensor = NULL;
	Tensor* xt_tensor = NULL;
	Tensor* rmse_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({f, n}),
                                                     &thetat_tensor));
	OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({f, m}),
                                                     &xt_tensor));
	OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({1, 1}),
													&rmse_tensor));

    auto thetat = thetat_tensor->template flat<float>();
    auto xt = xt_tensor->template flat<float>();
	auto rmse = rmse_tensor->template flat<float>();
	//initiate feature vector
	float* thetat_array = thetat.data();
	float* xt_array = xt.data();
	for (int k = 0; k < n * f; k++)
		thetat_array[k] = 0.1*((float) rand() / (float)RAND_MAX);
	//CG needs to initialize X as well
	for (int k = 0; k < m * f; k++)
		xt_array[k] = 0;

	// Call the cuda kernel launcher
	const int size = cscrow.size();
	printf("cscrow size: %d \n", size);
    //DoALSKernelLauncher(csrrow.data(), csrcol.data(), csrval.data(), thetat_array, xt.data(), N, nnz);
	rmse(0) = doALS(csrrow.data(), csrcol.data(), csrval.data(), cscrow.data(), csccol.data(), cscval.data(),
		coorow.data(), thetat_array, xt_array,
		coorowtest.data(), coocoltest.data(), coovaltest.data(),
		m, n, f, nnz, nnztest, lambda,
		iters, xbatch, thetabatch, deviceid);

  }
  //a void doALS, for testing purpose
  float doALS2(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const float* csrValHostPtr,
		const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const float* cscValHostPtr,
		const int* cooRowIndexHostPtr, float* thetaTHost, float * XTHost,
		const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const float * cooValHostTestPtr,
		const int m, const int n, const int f, const long nnz, const long nnz_test, const float lambda,
		const int ITERS, const int X_BATCH, const int THETA_BATCH, const int DEVICEID){
			return 1;
		}
	void DoALSKernelLauncher(const int* csrrow, const int* csrcol, const float* csrval, float* thetat, float* xt, 
	const int N, const int nnz) {
	  printf("in kernel\n");
	  for (int i = 0; i < N; i++) {
	  thetat[i] = 2 - csrrow[i];
	  xt[i] = csrrow[i]*2;
	  }
	}

};

REGISTER_KERNEL_BUILDER(Name("DoAls").Device(DEVICE_CPU), DoAlsOp);