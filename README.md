# A Practical Theory of Generalization in Selectivity Learning
- In this paper, we first demonstrate that _**a hypothesis class is learnable if predictors are induced by signed measures**_.  More importantly, we establish, **_under mild assumptions, that predictors from this class exhibit favorable OOD generalization error bounds_**. 

- Our theoretical results  allow us to design two improvement strategies for existing query-driven models: 1) **NeuroCDF**, that models the underlying cumulative distribution functions (CDFs) instead of the ultimate query selectivity; 2) a general training framework (**SeConCDF**) to enhance OOD generalization, without compromising good in-distribution generalization with any loss function. 

- This repository includes the codes, queries and scripts for our paper: A Practical Theory of Generalization in Selectivity Learning.

## Experiments of NeuroCDF
This section contains the experiments over a synthetic dataset from a highly-correlated 10-dimensional Gaussian distribution. 

We provide the codes for **LW-NN** (MSCN shares the same results with LW-NN in this section, and we will focus more on MSCN in next section), **LW-NN+NeuroCDF** (LW-NN trained with NeuroCDF), and **LW-NN+SeConCDF** (LW-NN trained with SeConCDF). 

The motivational experiments are minimal and clearly demonstrate 1) the superiority  of NeuroCDF over LW-NN on OOD generalization; and 2) the effectiveness of CDF self-consistency training of SeConCDF in enhancing LW-NN's generalization on OOD queries. 

Below are steps to reproduce the experiments. 

1. Please enter the [synthetic_data](synthetic_data) directory.
2. To reproduce the result of LW-NN, please run
```shell
    $ python train_LW-NN.py
```
3. To reproduce the result of LW-NN+NeuroCDF, please run
```shell
    $ python train_LW-NN+NeuroCDF.py
```
4. To reproduce the result of LW-NN+SeConCDF, please run
```shell
    $ python train_LW-NN+SeConCDF.py
```
5. Notice the RMSE and Median Qerror for both types  (In-Distribution and Out-of-Distribution) of test workloads across each model and compare the results!

## Experiments of SeConCDF
This section contains the experiments related to **SeConCDF**. You'll need a GPU for this section.

We first focus on **single-table** (Census) experiments. Below are steps to reproduce the experimental results.

1. Please enter the [single_table](single_table) directory.
2. To reproduce the result of LW-NN, please run
```shell
    $ python train_LW-NN
```
3. To reproduce the result of LW-NN+SeConCDF (LW-NN trained with SeConCDF), please run
```shell
    $ python train_LW-NN+SeConCDF
``` 

4. Similarly, run `train_MSCN.py` and `train_MSCN+SeConCDF.py` to reproduce the experiments of MSCN and MSCN+SeConCDF (MSCN trained with SeConCDF)


Then, we move on to **multi-table**  experiments. Below are steps to reproduce the experimental results.
1. Please enter the [multi_table](multi_table) directory.
2. Due to the file size limit of Github, please download the zipped directory of bitmaps from [this link](https://drive.google.com/file/d/1eBd4SJg8i8h9yv-dKDj-8ffWWqOyL9Qi/view?usp=sharing) and unzip it in current directory.
3. Similarly, download the zipped directory of workloads from [this link](https://drive.google.com/file/d/11dx95AXbAixgpHqcCajiq1TZN-n6F_wn/view?usp=sharing) and unzip it in current directory.
4. To reproduce the results on IMDb, please run
```shell
    $ python train_MSCN_imdb --shift granularity
    $ python train_MSCN+SeConCDF_imdb --shift granularity
```

where the parameter `shift` controls the OOD scenario (`granularity` for granularity shift and `center` for center move).


```shell
    $ python train_MSCN_imdb --shift center
    $ python train_MSCN+SeConCDF_imdb --shift center
```
You will observe substantial improvements (in terms of both RMSE and Q-error) with MSCN+SeConCDF compared to MSCN on OOD queries, which showcases the advantages of SeConCDF in multi-table cases.

5. Also, you can run `train_MSCN_dsb.py` and `train_MSCN+SeConCDF_dsb.py` to reproduce the experiments on DSB.


## Experiments with **SeConCDF** on **CEB-1a-varied**

This section describes the steps required to reproduce the experimental results of **SeConCDF** on the **CEB-1a-varied** workload. A GPU is also required for this section.


#### 1. Set Up the CEB-1a-varied Directory

Navigate to the [ceb-1a-varied](CEB-1a-varied) directory:


#### 2. Download and Extract the Workload

Due to GitHub's file size limits, the **ceb-1a-varied** workload must be downloaded separately. Follow these steps:

- Download the zipped directory from [this link](https://drive.google.com/file/d/1y_OUoiwbZPvboPR-kFC4xg9Ue0KPhQ58/view?usp=sharing).
- Extract the directory (named ceb-1a-varied-queries) under the [ceb-1a-varied](CEB-1a-varied) directory.

#### 3. Download and Load CSV Tables

The experiments require a set of CSV tables for bitmap computation. Follow these steps:

- Download the tables from [this link](https://drive.google.com/file/d/1V1hRv4XaWkp2ErPxyI9zY0HzJcS2Vh0l/view?usp=sharing).
- Load the tables into your local **PostgreSQL** database.
- Specify the connection parameters in the `connect_pg()` function within the following file:
  ```sh
  ceb-1a-varied/mscn/query_representation/generate_bitmap.py
  ```
  This ensures that the program can locate the sampled tables required for bitmap computation in MSCN.

#### 4. Use Preprocessed Workload (Optional, Recommended)

Preprocessing the workload can be slow. To save time, follow these steps:

- Download the preprocessed `.pkl` file from [this link](https://drive.google.com/file/d/1xnBv3N8RizaJTHB1_CyJJFrwoxu9jObE/view?usp=sharing).
- Extract the file and place it inside the [ceb-1a-varied/mscn](CEB-1a-varied/mscn)  directory. This will significantly reduce preprocessing time.

#### 5. Train and Evaluate Models
To reproduce the experimental results, return to the [ceb-1a-varied](CEB-1a-varied) directory and run the following commands:

```sh
python train_MSCN_ceb_1a-varied.py
python train_MSCN+SeConCDF_ceb-1a-varied.py
```

This will execute the training procedures necessary to reproduce the results of **SeConCDF** on **CEB-1a-varied** (compared to original MSCN).

