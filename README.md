Structure for this space : 


Experimented optimization : 

1.Hardware/GPU used : 

2.Before and After Optimization Results : 


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


************************************************************************************************************************************************************************************************
