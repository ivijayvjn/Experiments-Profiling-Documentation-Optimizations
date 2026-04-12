Structure for this space : 


Experimented optimization : 

1.Hardware/GPU used : 

2.Before and After Optimization Results : 

Lessons Learnt : 


************************************************************************************************************************************************************************************************

### <mark> Experimented optimization : 1.Kernel Fusion(Eager vs Torch.compile) </mark> 

a.Hardware/GPU used : NVIDIA RTX A5000(Ampere Architecture) : https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

b.Before and After Optimization Results : 


| Metric | Baseline (Eager) | Optimized(Fused) | Observation |
| :---: | :---: | :---: | :---: |
| Average Latency | 12.57 ms | 6.35 ms | ~1.98X speedup |
| Total VRAM Traffic | 8.56 GB | 2.14 GB | 75% Reduction |
|Peak Memory Usage | 2.15 GB | 1.07 GB | 50% Memory save |
|Arithmetic Intensity | 0.125FLOP/B | 0.500FLOP/B | 4X Efficiency |


**[View Full Report](./01.Kernel_Fusion)**

Lessons Learnt : 

This will be a lengthy one as I'll be recording all my experiences here as this is my first experiment documentation : 

1.Getting the Rented GPU cloud instance doesnt mean we get all the necessary control over it : 

a.ncu limitation : (A Tool I was so excited to work with for the speed of light, the Roofline model and especially the Memory map)

![ncu error](/./01.Kernel_Fusion/Assets/ncu_error.png)

b.linux perf limitation : (I wanted to see the difference in how CPU handles instructions and all beetween both the implementations)

![ncu error](/./01.Kernel_Fusion/Assets/perf_error.png)

However,We need to infer based on what data we have in our hand not guessing it but evaluating other metrics that we're having with us.

2.Just because fusing the code line into single operation doesnt mean you'll achieve kernel fusion magic but understanding the python's eager mode is important as torch.compile worked on its way here 

3.This whole profiling and documentaion took me whole day for understanding and observing this hardware magic but it never made me tired and I loved it

************************************************************************************************************************************************************************************************


### <mark> Experimented optimization : 2.KV-Cache characterization and Performance : Latency,Complexity and Power Efficiency </mark>

1.Hardware/GPU used : NVIDIA RTX A5000(Ampere Architecture) : https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

b.Before and After Optimization Results : 

| Metric(at 100 Tokens) | Baseline (No-cache) | Optimized(KV cache) | Improvement | 
|:--- | :--- | :--- | :--- |
|Average TPOT | 31.07 ms | 25.09 ms | 19.2% faster |
| P99 TPOT | 38.81 ms | 26.90 ms | 30.6 % stable |
| Peak Power(TDP) | 213 W | 171 W  | 42W(19.7%) savings | 
| Energy Profile | Compute-Bound| Memory-Bound | Efficiency shift |

**[View Full Report](./2.Inference_Scaling_KV_Cache_Efficiency)**

Lessons Learnt :

-Everytime I use LLMs I could see the stuttering of Tokens other than the first token and sometimes I've been the unlucky user of having the output tokens delayed.Right now I guess I was sitting at the p99 may be🤣🤣

-Whenever I come across the memory requirement of AI I always wanted to understand why so I wanted to do this profiling personally to check how this benefits AI workloads even though this is memory bound and Now I can understand that "There is nothing called extreme perfection you always need to find a balance by compromising something which is more efficient that one another

-Also KV reuse is not affecting the TTFT by any means as it is not helping the prefill phase that much may be thats why we prefer chunked prefilling,flash attention like methods are there.


************************************************************************************************************************************************************************************************

### <mark> Experiment : 3.Observability of gpt-2 using pytorch profiler,NVTX,Nvidia Nsight systems and perfetto </mark>

1.Hardware/GPU used : NVIDIA RTX A5000(Ampere Architecture) : https://github.com/ivijayvjn/Experiments-Profiling-Documentation-Optimizations/tree/d3473a3b6846b0b5d9e879595371242825cbab21/3.Diagnostic%20using%20Pytorch%20profiler%2CNVTX%20and%20Nsight%20Suite

2.Results/Understanding : 

***Disclaimer : Intentionally used a smaller batch***

2a.Software - Level Transparency(Pytorch Profiler & NVTX) : By using the NVTX markers hook,we are able to gain the logical observability.We can see exactly which python function correspond to GPU activities,transforming the unknown side of the training loop into a labeled,chronological sequence of events

2b.Perfetto Trace inspection : Perfetto provides dispatch observability as it allows us to measure microscopic view of the timeline and have show us how launch kernel was having higher number of calls which in turns inferred the launch overhead 

2c.System wide : Nsys captured the ground truth of the hardware.The 64.2 % cudakernellaunch statistic in the nsight summary proved that the system was launch bound.This confirmed that the CPU spent the majority of its time on administrative overhead rather than actual computation.

Lessons Learnt : I wanted understand the observability stack and how they work in to show us the complete skeleton of how the hardware works and how we can identify the bottlenecks for optimization..I wanted to understand these tools because the most importatnt part of optimization is to find the bottleneck with evidence so then we can optimize without doing any guess work or doing anti methods always mentioned by brendan gregg

**[View Full Report](./3.Diagnostic%20using%20Pytorch%20profiler,NVTX%20and%20Nsight%20Suite)**
