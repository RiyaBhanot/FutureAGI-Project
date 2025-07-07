import json
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(prompt):
    return f"""You are a math student who makes occasional reasoning mistakes.

Given the problem below, write a **flawed step-by-step solution** that seems reasonable but includes 1–2 subtle mistakes.

Problem:
{prompt}

Respond only with the flawed steps."""

def build_scoring_prompt(steps):
    formatted = "\n".join(f"- Step {i+1}: {step}" for i, step in enumerate(steps))
    return f"""You are a math grader. Rate each step below from 1 to 5 (higher = better).

Steps:
{formatted}

Return only a Python list of integers like: [3, 4, 2]"""

def ask_gpt(prompt):
    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT error: {e}")
        return None

def score_steps(steps):
    prompt = build_scoring_prompt(steps)
    response = ask_gpt(prompt)
    try:
        return json.loads(response)
    except:
        print("⚠️ Failed to parse scores:", response)
        return [1] * len(steps)  # fallback

def main(scored_path="data/scored_examples.jsonl", output_path="data/dpo_pairs_gpt.jsonl", limit=100):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(scored_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break

            example = json.loads(line)
            prompt = example["prompt"]
            chosen = example["completion"]
            scores_chosen = example["scores"]

            flawed_prompt = build_prompt(prompt)
            rejected_text = ask_gpt(flawed_prompt)
            if not rejected_text:
                print(f"⚠️ Skipping {i} due to GPT error.")
                continue

            rejected = [step.strip() for step in rejected_text.split("\n") if step.strip()]
            scores_rejected = score_steps(rejected)

            if len(scores_rejected) != len(rejected):
                print(f"⚠️ Step count mismatch at {i}. Using dummy scores.")
                scores_rejected = [1] * len(rejected)

            pair = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "scores_chosen": scores_chosen,
                "scores_rejected": scores_rejected
            }

            fout.write(json.dumps(pair) + "\n")
            print(f"✅ [{i}] Rejected + scores done.")

    print(f"\n🎉 Done! Saved to {output_path}")

if __name__ == "__main__":
    main()