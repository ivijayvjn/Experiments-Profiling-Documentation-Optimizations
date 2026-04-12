#importing model libraries and profiler libraries
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch import profiler

model_name = "openai-community/gpt2"
#tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda")
#downloading the model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#optimizer setup with fuison
optimizer = torch.optim.AdamW(model.parameters(),lr = 0.0001 , fused=True)

batch_size = 4
input_texts = ["IPL is a Huge Tournament because" , "IPL is better than " , "MI and CSK are successful because " , "IPL is costly"] 
#changing to pytorch tensor with padding and truncation
enc = tokenizer(input_texts,return_tensors = "pt" , padding = True , truncation = True)
input_ids = enc.input_ids.to(device)
#attention mask so to ignore padded values meaning consider only 1's not 0's
attention_mask = enc.attention_mask.to(device)
labels = input_ids.clone()

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
        
#results as json and also printing the required fields       
prof.export_chrome_trace("trace_results.json")
print(
        prof.key_averages().table(
                sort_by="self_cuda_time_total",
                                row_limit= 10,
                                max_name_column_width=35
                                )
        )





