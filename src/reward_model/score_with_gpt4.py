import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(question, steps):
    step_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    return f"""You are an expert math grader. Rate each reasoning step from 1 (completely incorrect) to 5 (completely correct).

Problem:
{question}

Steps:
{step_list}

Respond only in this JSON format:
{{"scores": [int, int, ...]}}"""

def score_with_gpt4(prompt):
    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = res.choices[0].message.content
        json_str = content[content.index("{"):content.rindex("}")+1]
        return json.loads(json_str)
    except Exception as e:
        print(f"❌ GPT-4 error: {e}")
        return {"scores": []}

def main(input_path="data/stepwise_prompts.jsonl", output_path="data/scored_examples.jsonl", limit=100):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break

            data = json.loads(line)
            prompt = build_prompt(data["prompt"], data["completion"])
            scores = score_with_gpt4(prompt)

            if scores["scores"]:
                data["scores"] = scores["scores"]
                fout.write(json.dumps(data) + "\n")
                print(f"✅ [{i}] Scored and saved.")
            else:
                print(f"⚠️ [{i}] Failed to score.")

if __name__ == "__main__":
    main()
