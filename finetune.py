from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
import torch
import torch.xpu  # Added for Intel GPU support
import intel_extension_for_pytorch as ipex  # Required for Intel GPU support
import os

MODEL_PATH = "./models/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_PATH = "./fine-tuned/tinyllama-lora"

os.environ["HF_HUB_OFFLINE"]="1"

# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("xpu")  # Changed from "cuda" to "xpu"

device = model.device

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)


# Load combined dataset (docs + QA)
def load_and_merge_datasets():
    docs = load_dataset("json", data_files="data/documents.jsonl")["train"]
    qas = load_dataset("json", data_files="data/qa_pairs.jsonl")["train"]

    def qa_format(example):
        return {"text": f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"}

    qas = qas.map(qa_format)
    all_data = Dataset.from_dict({"text": docs["text"] + qas["text"]})
    return all_data

dataset = load_and_merge_datasets()

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model (safetensors format)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)
