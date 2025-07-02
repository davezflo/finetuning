from transformers import AutoModelForCausalLM, AutoTokenizer

# Save model locally
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
save_path = "./models/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
