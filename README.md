# CuMF: CUDA-Acclerated ALS on mulitple GPUs. 

## What is matrix factorization?

Matrix factorization (MF) factors a sparse rating matrix R (m by n, with N_z non-zero elements) into a m-by-f and a f-by-n matrices, as shown below.

<img src=https://github.com/wei-tan/CuMF/raw/master/images/mf.png width=444 height=223 />
 
Matrix factorization (MF) is at the core of many popular algorithms, e.g., [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), word embedding, and topic model. GPU (graphics processing units) with massive cores and high intra-chip memory bandwidth sheds light on accelerating MF much further when appropriately exploiting its architectural characteristics.

## What is cuMF?

**CuMF** is a CUDA-based matrix factorization library that optimizes alternate least square (ALS) method to solve very large-scale MF. CuMF uses a set of techniques to maximize the performance on single and multiple GPUs. These techniques include smart access of sparse data leveraging GPU memory hierarchy, using data parallelism in conjunction with model parallelism, minimizing the communication overhead among GPUs, and a novel topology-aware parallel reduction scheme.

With only a single machine with four Nvidia GPU cards, cuMF can be 6-10 times as fast, and 33-100 times as cost-efficient, compared with the state-of-art distributed CPU solutions. Moreover, cuMF can solve the largest matrix factorization problem ever reported yet in current literature. 

CuMF achieves excellent scalability and performance by innovatively applying the following techniques on GPUs:  

(1) On one GPU, MF deals with sparse matrices, which makes it difficult to utilize GPU's compute power. We optimize memory access in ALS by various techniques including reducing discontiguous memory access, retaining hotspot variables in faster memory, and aggressively using registers. By this means cuMF gets closer to the roofline performance of a single GPU. 

(2) On multiple GPUs, we add data parallelism to ALS's inherent model parallelism. Data parallelism needs a faster reduction operation among GPUs, leading to (3).

(3) We also develop an innovative topology-aware, parallel reduction method to fully leverage the bandwidth between GPUs. By this means cuMF ensures that multiple GPUs are efficiently utilized simultaneously.

## Use cuMF to accelerate Spark ALS

CuMF can be used standalone, or to accelerate the [ALS implementation in Spark MLlib](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala).

We modified Spark's ml/recommendation/als.scala ([code](https://github.com/wei-tan/SparkGPU/blob/MLlib/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala)) to detect GPU and offload the ALS forming and solving to GPUs, while retain shuffling on Spark RDD. 

<img src=https://github.com/wei-tan/CUDA-MLlib/raw/master/als/images/spark-gpu.png width=380 height=240 />

This approach has several advantages. First, existing apps relying on mllib/ALS need no change. Second, we leverage the best of Spark (to scale-out to multiple nodes) and GPU (to scale-up in one node). Check this GitHub [project](https://github.com/wei-tan/CUDA-MLlib/tree/master/als) for more details.

## Build

Type:

	make clean build

To see debug message, such as run-time in each step, type:

	make clean debug

## Input Data

CuMF need training and testing rating matrices in binary format, and in CSR, CSC and COO formats. In ./data/netflix and ./data/ml10M we have already prepared (i)python scripts to download Netflix and Movielens 10M data, and preprocess them, respectively.

For Netflix data, type:

	cd ./data/netflix/
	python ./prepare_netflix_data.py 

Note: this can take 30+ minutes. You can download this [file](https://ibm.box.com/s/5vmh77up8reodvihiq0ri66jltg9h4uh) from your brower, extract and put the extracted files in ./data/netflix directly.

For Movielens:

	cd ./data/ml10M/
	ipython prepare_ml10M_data.py

Note: you will encounter a NaN test RMSE. Please refer to the "Known Issues" Section.

## Run

Type ./main you will see the following instructions:

Usage: give M, N, F, NNZ, NNZ_TEST, lambda, X_BATCH, THETA_BATCH and DATA_DIR.

E.g., for netflix data set, use:

	./main 17770 480189 100 99072112 1408395 0.058 1 3 ./data/netflix/
	
E.g., for movielens 10M data set, use:

	./main 71567 65133 100 9000048 1000006 0.05 1 1 ./data/ml10M/
	
E.g., for yahooMusic data set, use:

	./main 1000990 624961 100 252800275 4003960 1.1 6 3 ./data/yahoo/

Prepare the data as instructed in the previous section, before you run.

Note: rank value F has to be a multiply of 10, e.g., 10, 50, 100, 200). 

## Known Issues
We are trying to improve the usability, stability and performance. Here are some known issues we are working on:

(1) NaN test error. This is because in some data sets such as movielens 10M, there are users or items with no ratings in training set but some ratings in test set. To overcome this, we have defined a flag in als.cu (#define SURPASS_NAN). If SURPASS_NAN is defined, we check NaN in calculating RMSE and ignore the NaN values. Normally #define SURPASS_NAN should be commented out, as the additional check slows down the computation.

(2) Multi GPU support. We have tested on very large data sets such as [SparkALS](https://databricks.com/blog/2014/07/23/scalable-collaborative-filtering-with-spark-mllib.html) and HugeWiki, on multiple GPUs on one server. We will make our multi GPU support code available soon.

## References

More details can be found at:

1) Accelerate Recommender Systems with GPUs. Nvidia ParallelForAll [blog] ( https://devblogs.nvidia.com/parallelforall/accelerate-recommender-systems-with-gpus/).


2) CuMF: Large-Scale Matrix Factorization on Just One Machine with GPUs. Nvidia GTC 2016 talk. [ppt](http://www.slideshare.net/tanwei/s6211-cumf-largescale-matrix-factorization-on-just-one-machine-with-gpus), [video](http://on-demand.gputechconf.com/gtc/2016/video/S6211.html)

3) Faster and Cheaper: Parallelizing Large-Scale Matrix Factorization on GPUs. Wei Tan, [Liangliang Cao](https://github.com/llcao), [Liana Fong](https://github.com/llfong). [HPDC 2016], Kyoto, Japan. [(arXiv)](http://arxiv.org/abs/1603.03820)
