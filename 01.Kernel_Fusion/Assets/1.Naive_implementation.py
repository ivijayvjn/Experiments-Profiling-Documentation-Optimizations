import torch
import time 
from ctypes import cdll

#profiler_setup 
try:
    libcudart = cdll.LoadLibrary("libcudart.so")
except OSError:
    libcudart = cdll.LoadLibrary("libcudart.so.11")
#profiler_calls   
def profiler_start():
    libcudart.cudaProfilerStart()
    
def profiler_stop():
    libcudart.cudaProfilerStop()
    
#kernel_implementation 
def naive_operation(x):
    kernel1 = x + 1.0
    kernel2 = torch.relu(kernel1)
    kernel3 = kernel2 * 2.0
    kernel4 = torch.sigmoid(kernel3)
    return kernel4
#Experiment
def run_bench():
    x = torch.randn(16384 , 16384 , device="cuda")
    print(f"Experiment conducted on {torch.cuda.get_device_name(0)}...")
    
    #warmup 
    for _ in range(20):
        _ = naive_operation(x)
#synchronize to avoid python to move next before the GPU finishes        
    torch.cuda.synchronize()
    print("Ending the warmup..")

#profilers to track these iterations only     
    profiler_start()
    for _ in range(5):
        _ = naive_operation(x)
    torch.cuda.synchronize()
    profiler_stop()
    print("profiling range done")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range (100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]


#starting the 100 iterations here 
    print("Starting the 100 iterations...")
    for i in range(100):
        start_events[i].record()   
        _ = naive_operation(x)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s , e in zip(start_events , end_events)]
    avg_latency = sum(times) / len(times)

    print(f"Baseline Average latency : {avg_latency: .4f} ms")
    print(f"Total GPU memory : {torch.cuda.memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    run_bench()