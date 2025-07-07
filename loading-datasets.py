import json
from datasets import load_dataset
from pathlib import Path

def extract_examples(split="train", limit=100, out_path="data/stepwise_prompts.jsonl"):
    dataset = load_dataset("tasksource/PRM800K", split=split, streaming=True)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        count = 0
        for i, ex in enumerate(dataset):
            if count >= limit:
                break

            try:
                question = ex["question"]["problem"]
                raw_steps = ex["label"].get("steps", [])

                #Extract completions[].text from each step
                steps = []
                for step in raw_steps:
                    completions = step.get("completions", [])
                    if not isinstance(completions, list):
                        continue
                    for c in completions:
                        if "text" in c and c["text"].strip():
                            steps.append(c["text"].strip())
                            break  # only first completion

                if not question or not steps:
                    continue

                item = {
                    "prompt": question,
                    "completion": steps
                }

                f.write(json.dumps(item) + "\n")
                count += 1

            except Exception as e:
                print(f"Skipping {i} due to error: {e}")

    print(f"Saved {count} stepwise examples to {out_path}")

if __name__ == "__main__":
    extract_examples(split="train", limit=100)