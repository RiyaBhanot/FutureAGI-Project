## Areas Where LLM Was Used

###  1. **Data Scoring (Stepwise Rewards)**

- **LLM Used**: GPT-4 (via ChatGPT)
- **Purpose**: Generate stepwise token-level reward scores for `chosen` and `rejected` answers.
- **Method**: Prompted GPT-4 with pairs of step-by-step answers and asked it to assign token-level quality judgments.
- **Output Format**: JSONL with:
  - `prompt`
  - `chosen`, `rejected`
  - `scores_chosen`, `scores_rejected` (float list of same length as tokenized answer)

>  These outputs are stored in: `data/stepwise_prompts.jsonl`, `data/scored_examples.jsonl`

---

### 2. **Code Assistance**

#### 🔹 `train_dpo.py`
- Initial boilerplate generated via prompting ChatGPT to scaffold a HuggingFace `DPOTrainer` setup.
- Manual edits followed to match project-specific structure.

#### 🔹 `loading-datasets.py`
- Prompted GPT to write a script that loads JSONL files and converts them into HuggingFace `DatasetDict` format.
- Final implementation was custom-modified.

#### 🔹 `generate_from_model.py`
- Prompt-assisted script for model inference using `AutoModelForCausalLM` and `AutoTokenizer`.

---

### 3. **Documentation (README.md)**

- Prompted ChatGPT to generate an outline and professional structure.
- Included hand-written notes about project challenges and approach.

---

## Where LLM Was Not Used

- No model training, scoring logic, or critical code paths were fully automated.
- No plagiarism or blind copying from model outputs — all results manually reviewed and customized.
- No use of LLM in final reward aggregation logic (to be implemented manually in `StepwiseDPOTrainer`).

---

## Credits & Compliance

- The use of LLMs for assistance is explicitly allowed in the assignment.
- Every LLM-generated or guided component has been transparently declared here.
- This is in accordance with the evaluation criterion: **"Transparency – Clear attribution of any AI-generated content and thoughtful commit history."**

---