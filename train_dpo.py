from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig
from src.trainer.stepwise_dpo_trainer import StepwiseDPOTrainer
import torch

# === Load and Preprocess Dataset ===
print("🔹 Loading DPO dataset...")
dataset = load_dataset("json", data_files="data/dpo_pairs_converted.jsonl", split="train")

def format_example(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "reward_chosen": example["reward_chosen"],
        "reward_rejected": example["reward_rejected"]
    }

dataset = dataset.map(format_example)


# === Load Models ===
print("🔹 Loading SFT checkpoint...")
model_path = "./sft_model/checkpoint-147"
model = AutoModelForCausalLM.from_pretrained(model_path)
ref_model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


# === Training Config ===
training_args = DPOConfig(
    output_dir="./dpo_outputs",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=1e-5,
    fp16=torch.cuda.is_available(),
    beta=0.1,
    max_length=512,
    max_prompt_length=512,
    report_to="none"
)


# === Train ===
print("🔹 Starting DPO training...")
trainer = StepwiseDPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()