Experiment details

1.Experiment Title :

KV-Cache Scaling & Power Analysis

2.Objective :

Goal : 

To Quantify the overhead of KV-Cache scaling and power Analysis

Target Metrics :

2a.Shifting the complexity of decode phase from O(n^2) to O(1) by reusing the key value pairs as in KV caching.

2b.

3.Background / Hypothesis : 

1. KV Caching will reduce Decode latency by shifting the complexity from O(N^2) to O(1), making the process memory-bound.



4.System Environment : 
-CUDA version : 12.8

Driver Version : 570.124.06
-Pytorch Version : 2.7

-OS : Ubuntu 22.04.5

5.Measurement Methodology :

6.Baseline Measurement (Before Optimization) :
6a.Metric
6b.Nsight Observations :
7.Bottleneck Analysis :
8.Optimization Implemented :
9.Benchmark Procedure :
10.Results (After Optimization) :
11.Profiling Evidence :
12.Validation :
13.Tradeoffs :
14.Lessons Learned :
15.Future Work :


Commands Used : 

``` Bash
nsys --version #command not found

find / -name nsys

export PATH=$PATH:/opt/nvidia/nsight-compute/2024.3.2/host/target-linux-x64/

nsys --version

python static_batching_wo_kv_cache.py #test run to download the model

nvidia-smi -l 1

python static_batching_wo_kv_cache.py #10 token version

nvidia-smi -l 1

python static_batching_with_kv_cache.py #10 token version

nsys profile -o stat_batch_with_kv_cache_report python static_batching_wo_kv_cache.py

nsys profile -o stat_batch_with_kv_cache_report python static_batching_with_kv_cache.py

python static_batching_wo_kv_cache_Hundred.py #100 token version

python static_batching_with_kv_cache_hundred.py #100 token version
