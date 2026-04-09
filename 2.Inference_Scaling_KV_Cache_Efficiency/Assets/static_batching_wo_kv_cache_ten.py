import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np

device = "cuda"
model_id = "microsoft/Phi-3-mini-4k-instruct"

#using tokenizer to tokenize the input prompts and half precision  

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

#return as tensors and to device

prompts = ["The Name of Batman is " , "Explain Google's Dequant in detail"]

#padding to ensure both prompts are in equal matrices making sure as its static batching

inputs = tokenizer(prompts , return_tensors = "pt" , padding=True).to(device)
input_ids = inputs['input_ids']


#cuda_events_to_capture

start_event = torch.cuda.Event(enable_timing=True)
first_token_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

#start generation loop
generated_ids = input_ids 
token_events = []

#start time for ttft 

start_event.record()
# generating 10 tokens and passing the whole generated id every time :  without KV cache 
for i in range(100):
    with torch.no_grad():
        outputs = model(generated_ids, use_cache=False)
        
#get the last token's logits        
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits , dim=-1).unsqueeze(-1)
    
#append new token to the sequence 
    generated_ids = torch.cat([generated_ids , next_token], dim=-1)
    
#TTFT ends here 
    if i == 0:
        first_token_event.record()
        
#record every token further for TPOT timing 
    curr_event = torch.cuda.Event(enable_timing=True)
    curr_event.record()
    token_events.append(curr_event)
    
end_event.record()
torch.cuda.synchronize() #wait for GPU to finish so can read the times 

ttft = start_event.elapsed_time(first_token_event)
#calculate individual token durations to find p95,p99
durations = []
for i in range(len(token_events)):
    if i == 0:
        durations.append(start_event.elapsed_time(token_events[i]))
    else:
        durations.append(token_events[i - 1].elapsed_time(token_events[i]))
#p99,p95 and average
avg_tpot = np.mean(durations[1:])
p95 = np.percentile(durations[1:] , 95)
p99 = np.percentile(durations[1:] , 99)

print("The findings without KV cache for the input prompts as follows : ")
print(f'Time to first token : {ttft:.2f}ms')
print(f"Average Time per output token without KV caching : {avg_tpot:.2f}ms")
print(f"p95 TPOT : {p95:.2f} ms ")
print(f"p99 TPOT : {p99:.2f} ms ")
print(f"Total Sequence Length : {generated_ids.shape[1]} tokens")