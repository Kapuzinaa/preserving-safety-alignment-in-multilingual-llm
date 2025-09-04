import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import util
os.environ["CUDA_VISIBLE_DEVICES"] = str(util.get_free_gpu())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME = "google/gemma-2b-it"
M = MODEL_NAME.split("/")[1]
DATASET_NAME = "meta-math/MetaMathQA"
OUTPUT_DIR = "./fine-tuned-no-lora-" + M + "_20k"
BATCH_SIZE = 2
MAX_LENGTH = 512

def setup_model_and_tokenizer():
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer

def format_prompt(user_msg, assistant_msg):
        return f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"

def preprocess_dataset(dataset, tokenizer):
    """Preprocess the dataset for training"""
    def tokenize_function(examples):
        texts = [format_prompt(q, a) for q, a in zip(examples["original_question"], examples["response"])]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def main():
    print("Starting fine-tuning process...")
    
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    train_dataset = dataset["train"].select(range(20000))  # First 2000 samples
    
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    print("Preprocessing dataset...")
    tokenized_dataset = preprocess_dataset(train_dataset, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb logging
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
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
    test_question = "What is the derivative of x^2 + 3x + 5?"
    prompt = f"Question: {test_question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + 100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {prompt}")
    print(f"Output: {response}")

if __name__ == "__main__":
    
    main()