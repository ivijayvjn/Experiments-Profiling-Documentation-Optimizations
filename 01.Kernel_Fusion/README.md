1. Experiment Title : 

Improving arithmetic intensity and reducing memory overhead through "Kernel Fusion".

2. Objective : 

Goal : 

To Improve performance by reducing the CPU launch overhead,eliminating redundant memory round-trips to VRAM(Global Memory)

Target Metrics : 

2a.Arithmetic Intensity : Increase the ratio of FLOPs to bytes moved 

2b.Kernel : Reduce the total number of kernels executed for a single logical operation 

2c.Compute Throughput : Achieve a higher percentage of GPU's peak TFLOPS as in Memory bound --> Compute bound 

2d. Memory Bandwidth : Reduce unnecessary DRAM traffic for intermediate tensors 

3. Background / Hypothesis : 

3a. Operations that are executed sequentially forces the GPU to write back the intermediate results back to VRAM resulting the math cores to be idle as they wait for them to be read back again from the VRAM.This memory wall causes the math cores to be waiting for the data arrival resulting in the lower arithmetic intensity.Adding to it,each operation requires a kernel launch adding cumulative overhead.

3b. Expected Results :

Kernel Fusion will consolidate multiple operations into a single GPU pass.This will eliminate intermediate VRAM round-trips.Meaning the intermediate data stays in high speed registers/SRAM and we expect : 

-->"Increased Arithmetic Intensity" : The math operations will be more per byte of data moved from the global memory

-->"Higher Compute Throughput" : As the data is residing there for the next operation,we can see the math cores spending more time executing than waiting/stalled

-->"Latency Reduction" : Expecting the total execution time to be lesser as the kernel launches are decreased 

4. System Environment : 

-GPU : A5000Pro(Ampere Architecture)

-CUDA version : 12.8

- Driver Version : 570.124.06

-Pytorch Version : 2.7

-OS : Ubuntu 22.04.5

-Power/clock : Used default setting**

**"As the Cloud GPU instance provider denied the access to manually overturn the clock and power options,dropping the step to set it up and running the experiment with the default clock and power setting"


5. Measurement Methodology : 

Brendan Gregg's USE method 

Tools : 
-->nvidia-smi 
-->NVIDIA NSIGHT SYSTEMS(nsys)
-->NVIDIA NSIGHT COMPUTE(ncu)
-->linux perf 

6. Baseline Measurement (Before Optimization) : 

6a.Metric 

Baseline Performance Profile (Eager Mode)

| Metric | Technical Value | Rationale / Derivation |
| :--- | :--- | :--- |
| **Kernel Launches** | 4 Distinct | `add`, `relu`, `mul`, `sigmoid` |
| **Average Latency** | 12.57 ms | Measured via `torch.cuda.Event` (Synchronized) |
| **Peak VRAM Allocation** | 2.14 GB | $2 \times \text{Tensor Size}$ (Input + Materialized Intermediate) |
| **Total VRAM Traffic** | 8.56 GB | Calculated: $4 \times (1.07\text{GB Read} + 1.07\text{GB Write})$ |
| **Arithmetic Intensity** | 0.125 FLOP/B | Calculated: $1\text{ FLOP} / 8\text{ Bytes}$ (per element per kernel) |
| **Bus Utilization** | 88.6% | Measured Throughput (681 GB/s) / A5000 Peak (768 GB/s) |

![naive-Implementation](/./01.Kernel_Fusion/Assets/naive_commandline.png)

6b.Nsight Observations : 

Seperate Kernel launches recorded in Nsight Systems tool : 

![naive-Implementation](/./01.Kernel_Fusion/Assets/nsys_observation_naive.png)

7. Bottleneck Analysis : 

-As the GPU spends most of its time on VRAM I/O the operation chain is memory bound leading ti the math cores under-utilized leading to low Arithmetic intensity.So because of these memory round trips most of GPU's CUDA cores are under-utilized.

8. Optimization Implemented : 

BEFORE : 

Baseline code : 

![naive-Implementation](/./01.Kernel_Fusion/Assets/naive_code.png)

Partial_Fused_Code_Eager_mode : 

![partially_fused-Implementation](/./01.Kernel_Fusion/Assets/partial_fusion_code_eager.png)

Fully_fused_code_eager_mode : 

![fully_Implementation](/./01.Kernel_Fusion/Assets/Full_fusion_eager_code.png)

AFTER : 

Fully_fused_code_torch.compile_implementation : 

![torch.compile_fused_optimized](/./01.Kernel_Fusion/Assets/Kernel_fusion_torch_compile.png)

9. Benchmark Procedure : 

1.Warmup runs : 20 iterations 

2.Profiled runs : 5 iterations(Nsys)

3.Measured runs : 100 iterations(overall)

4.Same inputsize used 


10. Results (After Optimization) :

| Metric | Technical Value | Rationale / Derivation |
| :--- | :--- | :--- |
| **Kernel Launches** | 1 Distinct | Single-Triton generated pointwise kernel |
| **Average Latency** | 6.3465 ms | Measured via `torch.cuda.Event` (Synchronized) |
| **Peak VRAM Allocation** | 1.07 GB | $2 \times \text{Tensor Size}$ (Input + Materialized Intermediate) |
| **Total VRAM Traffic** |  2.14 GB | Calculated: $2 \times (1.07\text{GB Read} + 1.07\text{GB Write})$ |
| **Arithmetic Intensity** | 0.500 FLOP/B | Calculated: 4 * Increase via I/O elimination |
| **Bus Utilization** | 43.8% | Measured Throughput (337 GB/s) / A5000 Peak (768 GB/s) |

Terminal : 

![torch.compile_fused_optimized](/./01.Kernel_Fusion/Assets/Fused_op_result_commandline_torch_compile.png)

Nsys : 

![torch.compile_fused_optimized_nsys](/./01.Kernel_Fusion/Assets/Kernel_fused_torch.compile_nsys.png)

11.Profiling Evidence : 

All the Code,snippets,reports uploaded : https://github.com/ivijayvjn/Profiling-Experiments-Documentation/tree/acfffc7ab60b442418ffe50dd9fcd938780deaa6/01.Kernel_Fusion/Assets

12.Validation :

-Output Correctness verified against the code snippet

13.Tradeoffs : 

-Kernel fusion reduced VRAM traffic but it will surely put some pressure in SM resource utilization as the registers now have to carry the intermediate calculation

-Kernel fusion,in this case,havent yielded theoritical time drop which is 4* reduction from 12.57 ms to ~ 3 or 4 ms but 2 * speedup only (6.35 ms) which shows GPU is working on a different instruction taking extra time(may be sigmoid function here)

14.Lessons Learned : 

This will be a lengthy one as I'll be recording all my experiences here as this is my first experiment documentation : 

1.Getting the Rented GPU cloud instance doesnt mean we get all the necessary control over it : 

a.ncu limitation : (A Tool I was so excited to work with for the speed of light, the Roofline model and especially the Memory map)

![ncu error](/./01.Kernel_Fusion/Assets/ncu_error.png)

b.linux perf limitation : (I wanted to see the difference in how CPU handles instructions and all beetween both the implementations)

![ncu error](/./01.Kernel_Fusion/Assets/perf_error.png)

However,We need to infer based on what data we have in our hand not guessing it but evaluating other metrics that we're having with us.

2.Just because fusing the code line into single operation doesnt mean you'll achieve kernel fusion magic but understanding the python's eager mode is important as torch.compile worked on its way here 

3.This whole profiling and documentaion took me whole day for understanding and observing this hardware magic but it never made me tired and I loved it  

15.Future Work :

-Mixed precision and Quantization (Hopper or above)
   -->To analyse the trade-off between quantization error and hardware speedup
   
-CUDA Streams and TMA (Hopper or above)
   -->To implement and observe "COmpute and Communication Overlap"

   
-Continuous batching,Flash attention,Paged attention,KV caching
  -Chunked prefill and decode, Disintegrated prefill and decode,Speculative decoding with EAGLE,Medusa 
        -->To experiment on memory management and efficient TTFT,TPOT serving 

Commands Used : 

``` Bash

nvidia-smi 

sudo nvidia smi -pm 1

sudo nvidia-smi -pl 230

nvidia-smi -q -d=SUPPORTED_CLOCKS

sudo nvidia-smi -lgc 1410

nvidia-smi -q -d=CLOCK

cat /etc/os-release

python -c "import torch; print(torch.__version__)"

nsys --version #command not found error occured

find / -name nsys

export PATH=$PATH:/opt/nvidia/nsight-compute/2024.3.2/host/target-linux-x64/

ncu --version

perf --version #error occured as not found

apt-get install -y linux-tools-6.5.0-35-generic linux-cloud-tools-6.5.0-35-generic

sudo perf stat -e instructions,cycles python baseline.py  #Permission denied in cloud instance to access or record that event 

nvidia-smi --query-gpu=timestamp,pstate,utilization.gpu,utilization.memory,memory.used,clocks.current.sm,power.draw --format=csv -lms 100 -f naive_implementation_telemetry.csv

python 1.Naive_implementation.py

nsys profile -o naive_implementation_report -c cudaProfilerApi python 1.Naive_implementation.py

ncu --help # to find the --range-filter option

ncu --target-processes all --section SpeedOfLight -o naive_implementation_report_comp --range-filter :1:  python 1.Naive_implementation.py #The cloud provider is not providing the access to the counter to measure 

```
