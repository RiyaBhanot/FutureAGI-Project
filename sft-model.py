from datasets import load_dataset

def build_sft_format(example):
    return {
        "text": example["prompt"] + "\n" + example["chosen"]
    }

raw_dataset = load_dataset("json", data_files="data/dpo_pairs_converted.jsonl", split="train")
sft_dataset = raw_dataset.map(build_sft_format)

# Tokenize
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def tokenize(ex):
    encoded = tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded


tokenized = sft_dataset.map(tokenize)

# Train
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
import torch

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    trust_remote_code=True,
    use_safetensors=True
)


if torch.cuda.is_available():
    model = model.to("cuda")
    print("Model moved to GPU")
else:
    print("CUDA not available, training on CPU")

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./sft_model/",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch"
    )
)

trainer.train()

