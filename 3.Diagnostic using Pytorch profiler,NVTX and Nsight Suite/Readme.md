1. Experiment Title : 

Observability of GPT-2 Training using Pytorch Profiler,NVTX and Nsight Suite 

2. Objective : 

Using multilayered profiling tools,identify the primary architectural contsraints(Memory vs Compute bound) of a Casual LM training step and to characterize the baseline execution of the same 

3.Background / Hypothesis : 

Context : Modern deep learning training is not that transparent(claim as of my learning based on the resources available to me)so to observe the interaction between CPU and GPU the experiment is done 

Hypothesis : 

1.Pytorch profiler will reveal the code level calls
2.NVTX markers will reveal when an event is happened
3.Nsight systems will reveal the system wide calls completely stacked in the report and Nsight compute can reveal the skeleton of a single kernel completely 

Expected Outcome : 

Expect to observe the clear disctinction in the computational density between the forward and backward passes

4. System Environment : 

-GPU : A5000Pro(Ampere Architecture)

-CUDA version : 12.8

Driver Version : 570.124.06

-Pytorch Version : 2.7

-OS : Ubuntu 22.04.5

5.benchmark Procedure :

1.Warmup runs :  5 iterations

To ensure all CUDA kernels are compiled,the memory pool is populated and the GPU clocks have ramped up to their peak performance state 

2.Measured Iteration : a single iteration with batch size of 4

To ensure a cleantimeline without repeating too many overlapping patterns which makes the trace easier to read for learning purposes 

``` Python 

#warm-up not profiled
for _ in range(5):
    with torch.autocast(device_type="cuda",dtype = torch.bfloat16):
        outputs = model(input_ids,attention_mask = attention_mask , labels = labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

#setup profiler to track shapes,flops,stack
with profiler.profile(
    activities = [profiler.ProfilerActivity.CPU ,
                  profiler.ProfilerActivity.CUDA],
    record_shapes = True,
    profile_memory = True,
    with_stack = True,
    with_flops = True
) as prof:
    with profiler.record_function("train_step"):
#forward pass profiling
        torch.cuda.nvtx.range_push('forward')
        with torch.autocast(device_type="cuda" , dtype = torch.bfloat16):
            outputs = model(input_ids,attention_mask = attention_mask , labels = labels)
        loss = outputs.loss
#end of forward pass profiling
        torch.cuda.nvtx.range_pop()
#backward pass profiling
        torch.cuda.nvtx.range_push("Backward")
        loss.backward()
#optimizer step profiling
        torch.cuda.nvtx.range_push("Optimizer_step")
        optimizer.step()
#end of optimizer step profiling as nested profiling
        torch.cuda.nvtx.range_pop()
        optimizer.zero_grad()
#end of backward pass profiling as nested profiling
        torch.cuda.nvtx.range_pop()
        
prof.export_chrome_trace("trace_results.json")
print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                row_limit= 10,
                                max_name_column_width=35
                                ))
```

6.Baseline Observations : 

Terminal Output : 

![Terminal_output](/./3.Diagnostic%20using%20Pytorch%20profiler,NVTX%20and%20Nsight%20Suite/Assets/Terminal_output_result.png)

As per the output attached,observed that eager mode gets kick in and the higher number of calls for aten::mm and aten::layer_norm indicates a Kernel launch overhead bottleneck.The GPU is being provided many small requests rathen that being given a large efficient blocks of work.We've already done an experiment on Kernel fusion where we analysed how torch.compile executes graph and eliminates this launch overhead.This in turn time and time proved me illustrating why techniques like Kernel fusion and torch.compile are necessary to saturate high-performance GPUs

Perfetto/chrome tracing : 

![Perfetto_output](/./3.Diagnostic%20using%20Pytorch%20profiler,NVTX%20and%20Nsight%20Suite/Assets/perfetto_diagnostic_summary.png)

As per the above quantitative analysis perfetto pivot table,I can observe a higher degree of operator fragmentation.specifically,Cudalaunchkernel was invoked 992 times in a single iteration.The higher frequency of memory management ops (aten :: copy and aten :: to)relative to actual compute kernels confirms that the execution is launch latency limited.The CPU spends a significant amount of time managing 900+ small tasks rather than dispatiching fewer,larger,more efficient compute blocks


Nsys : 

![Nsys_output](/./3.Diagnostic%20using%20Pytorch%20profiler,NVTX%20and%20Nsight%20Suite/Assets/nsys_diagnostic_summary.png)


As per the above we can see the same observation that this is launch bound.As these each kernel is computationally thin,the overhead of launching the kernel outweighs the benefit of GPU acceleration.This explains why the A5000 is under utilized.The CPU cannot feed the GPU fast enough  

7.Profiling Evidence :

All the Code,snippets,reports uploaded : https://github.com/ivijayvjn/Experiments-Profiling-Documentation-Optimizations/tree/3b07413a4de4dbd4618ac53f1586a034882fd74f/3.Diagnostic%20using%20Pytorch%20profiler%2CNVTX%20and%20Nsight%20Suite

8.Lessons Learnt : 

1.Multi-Level Observability : Using the pytorch profiler with NVTX markers can map high level python logic to low level execution order,identyfying where the code structure creates overhead.

2.Visual Trace Analysis : Used perfetto to visualize execution timelines and quantify bottlenecls,specifically discovering that Cudalaunch kernel calls were dominating the CPU time.

3.Hardware-Level Diagnostics : Used NVIDIA Nsight systems for system wide telemetry,confirming that system wide statistics can be observed to find the bottlenecks 

Commands Used : 

```Bash

nvidia-smi

python 3.Diagnostics_using_pytorch_profiler_NVTX_nsight_suite.py

https://ui.perfetto.dev #safari user so used this instead of chrome tracing

find / -name nsys

export PATH=$PATH:/opt/nvidia/nsight-compute/2024.3.2/host/target-linux-x64

nsys profile -t nvtx,cuda -o gpt2_report python 3.Diagnostics_using_pytorch_profiler_NVTX_nsight_suite.py # recording only the nvtx marked area of the code 





