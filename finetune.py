import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import util
from trl import SFTTrainer, SFTConfig

from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = str(util.get_free_gpu())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# Configuration
MODEL_NAME = "google/gemma-2b-it"
#MODEL_NAME = "./frozen/Llama-3.2-1B-Instruct-frozen-new"
#MODEL_NAME = "./frozen/gemma-2b-it-frozen-new"
#MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
#MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
M = MODEL_NAME.split("/")[-1]
DATASET_NAME = "meta-math/MetaMathQA"
OUTPUT_DIR = "./models/fine-tuned-" + M
BATCH_SIZE = 8
MAX_LENGTH = 512

def setup_model_and_tokenizer():
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = model.to("cuda")
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    lora_config = LoraConfig(
        r=16,                    # Low rank
        lora_alpha=32,           # LoRA scaling parameter
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
        lora_dropout=0.05,        # LoRA dropout
        bias="none",             # No bias training
        task_type="CAUSAL_LM"    # Causal language modeling
    )
    return lora_config

def to_chat(tokenizer, data):
    return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": data["original_question"]},
                {"role": "assistant", "content": data["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    tokenized_dataset = dataset["train"].select(range(20000)).map(lambda d: {"text": to_chat(tokenizer, d)})  # First 20000 samples
    
    print("Applying LoRA configuration...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=10,
        max_seq_length=1024,
        save_steps=100,
        optim="paged_adamw_8bit",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=True,
        dataloader_pin_memory=False,
        report_to=None,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to {OUTPUT_DIR}")
    
    print("\nTesting the fine-tuned model...")
    test_inference(model, tokenizer)

def test_inference(model, tokenizer):
    q = "What is the derivative of x^2 + 3x + 5?"
    prompt = tokenizer.apply_chat_template(
        [{"role":"user","content": q}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    
    main()
