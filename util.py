import torch, os, GPUtil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,            # Use 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization (improves accuracy)
    bnb_4bit_quant_type='nf4',    # Options: 'fp4' or 'nf4' (nf4 is usually better)
    bnb_4bit_compute_dtype='bfloat16'  # Or 'float16', depending on hardware
)

def load_model(model_name):
    models = [
        "meta-llama/Llama-3.2-1B-Instruct", #0
        "meta-llama/Llama-3.2-3B-Instruct", #1
        "Qwen/Qwen2.5-7B-Instruct", #2
        "Qwen/Qwen2.5-14B-Instruct", #3
        "google/gemma-2b-it", #4
        "google/gemma-7b-it", #5
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", #6
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", #7
        "microsoft/Phi-4-mini-instruct", #8 different syntax
        "microsoft/Phi-4-mini-flash-reasoning", #9 different syntax
    ]
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_judge_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_free_gpu():
    # Get a list of all GPUs and their status
    gpus = GPUtil.getGPUs()
    # If there are no GPUs available, raise an error
    if not gpus:
        raise RuntimeError("No GPU available.")
    # Sort GPUs by available memory (descending order)
    gpus_sorted_by_memory = sorted(gpus, key=lambda gpu: gpu.memoryFree, reverse=True)
    # Select the GPU with the most free memory
    selected_gpu = gpus_sorted_by_memory[0]
    print(f"Selected GPU ID: {selected_gpu.id}")
    return selected_gpu.id

def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True
    )
    response = tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)
    return response

def get_jailbreak_dataset(dataset_name="JailbreakV-28K/JailBreakV-28k"):
    dataset = load_dataset(dataset_name, 'JailBreakV_28K')
    return dataset

def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset
