Experiment details

1.Experiment Title :

KV-Cache characterization and Performance : Latency,Complexity and Power Efficiency on NVIDIA Ampere Architecture

2.Objective :

Goal : 

TO conduct an analysis and compare standard autoregressive decoding and KV-Cached decoding,quantifying the operational overhead of cache management versus the computational savings in high token count generation

Target Metrics :

2a.Arithmetic Complexity : Shifting the complexity of decode phase from O(n^2) [reduntant recompuation] to O(1) per token complexity by reusing the key value states in GPU VRAM.

2b.Hardware Resource utilization and Thermal Profile : Quantify the reduction in tensor core utilization and the impact on GPU power draw power draw shifting the bottleneck from compute bound state to a memory-bound state

3.Background / Hypothesis : 

1. Enabling KV cache reuse will decouple the relationship between sequence length and compute time.Reuse of the historical keys and values transform the attention calculation from a quadratic O(n^2) operation into a constant-time O(1) lookup per new token,resulting in significantly lower Time per Output Token(TPOT) as context grows

2.We anticipate a measurable reduction in Thermal Design power(TDP) usage.Because the compuatational burden shifts from performing redundant matrix multiplications to the more energy-efficient VRAM memory bandwidth,the total GPU power draw will decrease, demonstrating a higher "performance-per-watt" ratio



4.System Environment : 
-CUDA version : 12.8

Driver Version : 570.124.06
-Pytorch Version : 2.7

-OS : Ubuntu 22.04.5

5.Measurement Methodology :

"Adapted RED method and the USE method for power draw" : 

1.Duration : Time to first token(TTFT) to measure prefill efficiency and Time per output token(TPOT) to measure the stability of the decode phase 
2.Utilization and Power draw : using the USE method,real time TDP(power draw) and VRAM footprint was captured at a 1 second sampling interval


6.Baseline Measurement (Before Optimization-Without KV pair reuse) : "STANDARD AUTOREGRESSIVE DECODING(NO KV caching)

``` Python :

# generating 10 tokens and passing the whole generated id every time :  without KV cache 
for i in range(10):
    with torch.no_grad():
        outputs = model(generated_ids, use_cache=False)

```
``` Python : 

# generating 100 tokens and passing the whole generated id every time :  without KV cache 
for i in range(100):
    with torch.no_grad():
        outputs = model(generated_ids, use_cache=False)
```

| Metric | Tokens Generated-10 | Tokens Generated-100 | Observation 
|:--- | :--- | :--- | :--- |
|Average TPOT | 23.36 ms | 31.07 ms | +33% Latency Increase |
| P99 TPOT | 24.01 ms | 38.81 ms | Tail Stutter |
| Peak Power(TDP) | 84 W | 213 W | +153% power surge | 
| GPU Utilization | 73% | 29% - 30% - 44% | Inefficiency |


6a.Metrics in Tool Observation : 

Command Line output : 

10 Tokens w/o KV reuse : 

![10_tokens_wo_kv_cache](/./2.Inference_Scaling_KV_Cache_Efficiency/Assets/Terminal_static_batching_wo_kv_cache_10.png)

100 Tokens w/o Reuse : 

![100_tokens_wo_kv_cache](/./2.Inference_Scaling_KV_Cache_Efficiency/Assets/Terminal_stat_batch_wo_kv_cache_hundred_tokens.png)

Nvidia-smi : 

10 Tokens w/o KV reuse : 

![10_tokens_wo_kv_cache](/./2.Inference_Scaling_KV_Cache_Efficiency/Assets/nvidia-smi_stat_batch_wo_kvcache_10.png)


100 Tokens w/o KV reuse : 

![100_tokens_wo_kv_cache](/./2.Inference_Scaling_KV_Cache_Efficiency/Assets/nvidia-smi_wo_kvcache_hundred_tokens.png)

7.Bottleneck Analysis :

Root Cause and Evidence : 

7a.Redundant Computation Dominance : 

Evidence : Terminal output shows 33% increase in TPOT from 23ms to 31ms as the sequence length grows from 10 to 100.It confirms math cores are occupied with ever increasing redundant operations to produce a single output 

7b . Power-utilization-problem : 

Evidence : Power draw at 92% of TDP(213 W) shows the tensor cores operating at high volate/frequency to handle heavy matrix multiplication load yet the GPU utilization is at only 44% shows its compute inefficient and burning maximum energy to re calculate known values while the actual throughput is throttled by overhead of refetching the entire prompt history from VRAM for every iteration

8.Optimization Implemented :

``` Python 

#change 1 : Past key values passed and the use cache set to be true
for i in range(100):
    with torch.no_grad():
        outputs = model(current_input_ids, 
                        past_key_values=past_key_values, 
                        use_cache=True
                        )
    past_key_values = outputs.past_key_values        
#get the last token's logits        
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits , dim=-1).unsqueeze(-1)
# sending only the new token for next loop    
    current_input_ids = next_token
```

9.Benchmark Procedure :

1.Warmup run : 1 iteration for the model to be downloaded and initial handshake between the python host and GPU device.Prevents first token latency spike from skewing steady state metrics

2.Measured runs : 3 controlled measured runs (in both 10 tokens and 100 tokens --> Baseline vs KV-Cache)

3.Results reported as an average across the measured runs with a specific focus on p95 and p99 

4.Input dataset(prompts and batch size were held constant

10.Results (After Optimization) :

Terminal Output with KV reuse 100 Tokens: 

![100_tokens_with_kv_cache](/./2.Inference_Scaling_KV_Cache_Efficiency/Assets/Terminal_static_batch_with_kv_hundred.png)

Nvidia-smi with KV reuse 100 Tokens : 

![100_tokens_with_kv_cache](/./2.Inference_Scaling_KV_Cache_Efficiency/Assets/nvidia-smi_statbatch_with_kv_hundred_token.png)


| Metric(at 100 Tokens) | Baseline (No-cache) | Optimized(KV cache) | Improvement | 
|:--- | :--- | :--- | :--- |
|Average TPOT | 31.07 ms | 25.09 ms | 19.2% faster |
| P99 TPOT | 38.81 ms | 26.90 ms | 30.6 % stable |
| Peak Power(TDP) | 213 W | 171 W  | 42W(19.7%) savings | 
| Energy Profile | Compute-Bound| Memory-Bound | Efficiency shift |


11.Profiling Evidence :

All the Code,snippets,reports uploaded : https://github.com/ivijayvjn/Experiments-Profiling-Documentation-Optimizations/tree/71e037a306da933e13132ab7727d160d6ff53225/2.Inference_Scaling_KV_Cache_Efficiency/Assets


12.Tradeoffs :

-Enabling KV reuse shifted the bottleneck from tensore core to VRAM bandwidth.while this reduces power and latency it makes the workload memory-bound..

-This is what results in a higher VRAM footprint.storing the KV tensors for every layer and every head consumes significant memory.This may result in limiting the batch size as we may face OOM error.

-As noted in the report utilization will remain low so to achieve true hardware saturation we might need to implement the continous batching where we can queue prefill inside the less used math cores

14.Lessons Learned :

-Everytime I use LLMs I could see the stuutering of Tokens other than the first token and sometimes I've been the unlucky user of having the output tokens delayed.Right now I guess I was sitting at the p99 may be🤣🤣

-Whenever I come across the memory requirement of AI I always wanted to understand why so I wanted to do this profiling personally to check how this benefits AI workloads even though this is memory bound and Now I can understand that "There is nothing called extreme perfection you always need to find a balance by compromising something which is more efficient that one another

-Also KV reuse is not affecting the TTFT by any means as it is not helping the prefill phase that much may be thats why we prefer chunked prefilling,flash attention like methods are there. 


15.Future Work :

-Worked on static batching made me think about continuous batching 
-KV cache means I need to check how paged attention efficiently handles the memory movement
-Also need to profile slab allocator vs stream ordered allocator(cudaMallocAsync) and its combination with paged attention


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
