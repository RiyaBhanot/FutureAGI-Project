from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Load DPO-trained model ===
model_path = "sft_model/checkpoint-49" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# === Prompt to test ===
prompt = "Q: If you flip a coin 3 times, what is the probability of getting at least one head?\nA:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Output ===")
print(response)
