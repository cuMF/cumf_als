# CuMF: CUDA-Acclerated ALS on mulitple GPUs. 

## What is matrix factorization?

Matrix factorization (MF) factors a sparse rating matrix R (m by n, with N_z non-zero elements) into a m-by-f and a f-by-n matrices, as shown below.

<img src=https://github.com/wei-tan/CuMF/raw/master/images/mf.png width=444 height=223 />
 
Matrix factorization (MF) is at the core of many popular algorithms, e.g., [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), word embedding, and topic model. GPU (graphics processing units) with massive cores and high intra-chip memory bandwidth sheds light on accelerating MF much further when appropriately exploiting its architectural characteristics.

## What is cuMF?

**CuMF** is a CUDA-based matrix factorization library that optimizes alternate least square (ALS) method to solve very large-scale MF. CuMF uses a set of techniques to maximize the performance on single and multiple GPUs. These techniques include smart access of sparse data leveraging GPU memory hierarchy, using data parallelism in conjunction with model parallelism, minimizing the communication overhead among GPUs, and a novel topology-aware parallel reduction scheme.

With only a single machine with four Nvidia GPU cards, cuMF can be 6-10 times as fast, and 33-100 times as cost-efficient, compared with the state-of-art distributed CPU solutions. Moreover, cuMF can solve the largest matrix factorization problem ever reported yet in current literature. 

CuMF achieves excellent scalability and performance by innovatively applying the following techniques on GPUs:  

(1) On a single GPU, MF deals with sparse matrices, which makes it difficult to utilize GPU's compute power. We optimize memory access in ALS by various techniques including reducing discontiguous memory access, retaining hotspot variables in faster memory, and aggressively using registers. By this means cuMF gets closer to the roofline performance of a single GPU. 

(2) On multiple GPUs, we add data parallelism to ALS's inherent model parallelism. Data parallelism needs a faster reduction operation among GPUs, leading to (3).

(3) We also develop an innovative topology-aware, parallel reduction method to fully leverage the bandwidth between GPUs. By this means cuMF ensures that multiple GPUs are efficiently utilized simultaneously.

## Use cuMF to accelerate Spark ALS

CuMF can be used standalone, or to accelerate the [ALS implementation in Spark MLlib](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala).

We modified Spark's ml/recommendation/als.scala ([code](https://github.com/wei-tan/SparkGPU/blob/MLlib/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala)) to detect GPU and offload the ALS forming and solving to GPUs, while retain shuffling on Spark RDD. 

<img src=https://github.com/wei-tan/CUDA-MLlib/raw/master/als/images/spark-gpu.png width=380 height=240 />

This approach has several advantages. First, existing Spark applications relying on mllib/ALS need no change. Second, we leverage the best of Spark (to scale-out to multiple nodes) and GPU (to scale-up in one node). Check this GitHub [project](https://github.com/wei-tan/CUDA-MLlib/tree/master/als) for more details.

## Build

Type:

	make clean build

To see debug message, such as run-time in each step, type:

	make clean debug

## Input Data

Users can prepare input to cuMF in text format, with each line like:

	item_id user_d rating
	
As an example, we can start from the netflix data from [here](http://www.select.cs.cmu.edu/code/graphlab/datasets/).
You can download netflix_mm and netflix_mme from the above [URL](http://www.select.cs.cmu.edu/code/graphlab/datasets/).

The netflix_mm and netflix_mme files look like

	% Generated 25-Sep-2011
	480189 17770 99072112
	1 1  3
	2 1  5
	3 1  4
	5 1  3
	6 1  3
	7 1  4
	8 1  3

Please refer to this python [script](https://github.com/wei-tan/CuMF/blob/master/scripts/prepare_input.ipynb) to prepare cuMF input data. 

## Run

For example, to run cuMF with netflix data set:
Run main, specifying F (rank value), lambda, and how many partitions on solving movie features (this is because netflix data has far more movies than users). Prepare the data as requested in the test-als.cu file. We will make this run script more user-friendly later.

Note: rank value has to be a multiply of 10, e.g., 10, 50, 100, 200). On K40 here is how we run:

	./main 100 0.058 3

## Known Issues
We are trying to improve the usability, stability and performance. Here are some known issues we are working on:

(1) More user-friendly data transformation and run scripts.

(2) Multi GPU support. We have tested on very large data sets such as [SparkALS](https://databricks.com/blog/2014/07/23/scalable-collaborative-filtering-with-spark-mllib.html) and HugeWiki, on multiple GPUs on one server. We will make our multi GPU support code available soon.

## References

More details can be found at:

1) Accelerate Recommender Systems with GPUs. Nvidia ParallelForAll [blog] ( https://devblogs.nvidia.com/parallelforall/accelerate-recommender-systems-with-gpus/).


2) CuMF: Large-Scale Matrix Factorization on Just One Machine with GPUs. Nvidia GTC 2016 talk. [ppt](http://www.slideshare.net/tanwei/s6211-cumf-largescale-matrix-factorization-on-just-one-machine-with-gpus), [video](http://on-demand.gputechconf.com/gtc/2016/video/S6211.html)

3) Faster and Cheaper: Parallelizing Large-Scale Matrix Factorization on GPUs. Wei Tan, [Liangliang Cao](https://github.com/llcao), [Liana Fong](https://github.com/llfong). [HPDC 2016], Kyoto, Japan. [(arXiv)](http://arxiv.org/abs/1603.03820)
