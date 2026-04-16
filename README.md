Structure for this space : 


Experimented optimization : 

1.Hardware/GPU used : 

2.Key Insights/Key findings : 

************************************************************************************************************************************************************************************************

### <mark> Experimented optimization : 1.Kernel Fusion(Eager vs Torch.compile) </mark> 

Objective :  

Improve arithmetic intensity and reduce memory overhead by consolidating multiple pointwise operations into a single fused kernel using torch.compile.

a.Hardware/GPU used : NVIDIA RTX A5000(Ampere Architecture) : https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

b.Before and After Optimization Results : 

| Metric | Baseline (Eager) | Optimized(Fused) | Observation |
| :---: | :---: | :---: | :---: |
| Average Latency | 12.57 ms | 6.35 ms | ~1.98X speedup |
| Total VRAM Traffic | 8.56 GB | 2.14 GB | 75% Reduction |
|Peak Memory Usage | 2.15 GB | 1.07 GB | 50% Memory save |
|Arithmetic Intensity | 0.125FLOP/B | 0.500FLOP/B | 4X Efficiency |

Key Insights : 

--> Kernel fusion significantly reduces DRAM traffic and kernel launch overhead.
--> Increased arithmetic intensity shifts the workload toward better compute utilization.
--> Cloud GPU environments may limit access to low-level profiling features.


**[View Full Report](./01.Kernel_Fusion)**

************************************************************************************************************************************************************************************************

### <mark> Experimented optimization : 2.KV-Cache characterization and Performance : Latency,Complexity and Power Efficiency </mark>

Objective : 

Characterize the impact of KV-cache reuse on latency, computational complexity, and power efficiency during autoregressive decoding.

1.Hardware/GPU used : NVIDIA RTX A5000(Ampere Architecture) : https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

b.Before and After Optimization Results : 

| Metric(at 100 Tokens) | Baseline (No-cache) | Optimized(KV cache) | Improvement | 
|:--- | :--- | :--- | :--- |
|Average TPOT | 31.07 ms | 25.09 ms | 19.2% faster |
| P99 TPOT | 38.81 ms | 26.90 ms | 30.6 % stable |
| Peak Power(TDP) | 213 W | 171 W  | 42W(19.7%) savings | 
| Energy Profile | Compute-Bound| Memory-Bound | Efficiency shift |

Key Insights : 

--> KV-cache reuse reduces redundant computation, improving latency stability.

--> The optimization shifts the workload from compute-bound to memory-bound.

--> While TPOT improves, TTFT remains largely unaffected, highlighting the need for techniques like FlashAttention or chunked prefilling.

**[View Full Report](./2.Inference_Scaling_KV_Cache_Efficiency)**

************************************************************************************************************************************************************************************************

### <mark> Experiment : 3.Observability of gpt-2 using pytorch profiler,NVTX,Nvidia Nsight systems and perfetto </mark>

Objective : Provide multi-layer observability of a GPT-2 training step using PyTorch Profiler, NVTX markers, and NVIDIA Nsight Systems to identify architectural bottlenecks

1.Hardware/GPU used : NVIDIA RTX A5000(Ampere Architecture) : https://github.com/ivijayvjn/Experiments-Profiling-Documentation-Optimizations/tree/d3473a3b6846b0b5d9e879595371242825cbab21/3.Diagnostic%20using%20Pytorch%20profiler%2CNVTX%20and%20Nsight%20Suite

2.Results/Understanding : 

Key Findings : 

--> Software-Level Transparency: NVTX markers map high-level Python operations to GPU activities.
--> Dispatch Observability: Perfetto traces reveal excessive kernel launches.
--> System-Level Insight: Nsight Systems identified the workload as launch-bound, with ~64% of time spent on cudaKernelLaunch.


Key insights : 

--> Operator fragmentation leads to significant kernel launch overhead.
--> Multi-layer profiling enables evidence-based optimization.
--> Observability is a prerequisite for effective performance tuning.


**[View Full Report](./3.Diagnostic%20using%20Pytorch%20profiler,NVTX%20and%20Nsight%20Suite)**

************************************************************************************************************************************************************************************************
