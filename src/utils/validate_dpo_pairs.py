import json

def validate_dpo_file(path="data/dpo_pairs_gpt.jsonl"):
    total = 0
    valid = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            try:
                ex = json.loads(line)

                prompt = ex.get("prompt", "")
                chosen = ex.get("chosen", [])
                rejected = ex.get("rejected", [])
                scores_chosen = ex.get("scores_chosen", [])
                scores_rejected = ex.get("scores_rejected", [])

                # Check presence
                if not prompt or not chosen or not rejected:
                    print(f"❌ Line {i}: Missing prompt/steps.")
                    continue

                # Length checks
                if len(chosen) != len(scores_chosen):
                    print(f"❌ Line {i}: chosen vs scores_chosen mismatch ({len(chosen)} vs {len(scores_chosen)})")
                    continue
                if len(rejected) != len(scores_rejected):
                    print(f"❌ Line {i}: rejected vs scores_rejected mismatch ({len(rejected)} vs {len(scores_rejected)})")
                    continue

                # Score value check (optional)
                if any(s not in [1,2,3,4,5] for s in scores_chosen + scores_rejected):
                    print(f"⚠️ Line {i}: Invalid score values (should be 1–5)")

                valid += 1

            except Exception as e:
                print(f"❌ Line {i}: JSON error: {e}")

    print(f"\n✅ {valid}/{total} examples are valid.")

if __name__ == "__main__":
    validate_dpo_file()
