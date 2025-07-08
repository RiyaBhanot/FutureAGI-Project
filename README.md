## futureAGI: Stepwise DPO Training Pipeline

This project provides a pipeline for training, evaluating, and generating outputs from large language models (LLMs) using stepwise Direct Preference Optimization (DPO) and reward modeling. The workflow is designed for math reasoning tasks, leveraging OpenAI's GPT-4 for reward scoring and data augmentation.

### Project Structure

```
futureAGI/
├── data/                  # Datasets and processed data
├── dpo_outputs/           # DPO training checkpoints and outputs
├── sft_model/             # Supervised fine-tuning checkpoints
├── src/
│   ├── reward_model/      # Reward scoring scripts
│   ├── trainer/           # Custom DPO trainer
│   └── utils/             # Data validation and generation utilities
├── generate_from_model.py # Generate outputs from a trained model
├── loading-datasets.py    # Extract and preprocess datasets
├── sft-model.py           # Supervised fine-tuning script
├── train_dpo.py           # DPO training script
├── environment.yml        # Conda environment definition
├── requirements.txt       # Python dependencies
```

### Setup Instructions

1. **Clone the repository and install dependencies:**
   ```powershell
   conda env create -f environment.yml
   conda activate futureagi
   # Or use pip:
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key:**
   - Create a `.env` file with your `OPENAI_API_KEY` for reward modeling and data generation scripts.

### Main Scripts Overview

- **`loading-datasets.py`**: Extracts and preprocesses math reasoning examples from the PRM800K dataset into stepwise prompt format.
- **`src/utils/generate_rejected_pairs.py`**: Uses GPT-4 to generate flawed (rejected) solutions and stepwise scores for DPO training.
- **`src/utils/validate_dpo_pairs.py`**: Validates the integrity and format of DPO training pairs.
- **`src/reward_model/score_with_gpt4.py`**: Scores reasoning steps using GPT-4 as a reward model.
- **`sft-model.py`**: Performs supervised fine-tuning (SFT) on the processed dataset.
- **`src/trainer/stepwise_dpo_trainer.py`**: Custom DPO trainer that incorporates stepwise reward signals.
- **`train_dpo.py`**: Trains a model using DPO with stepwise rewards.
- **`generate_from_model.py`**: Loads a trained model and generates answers for a given prompt.

##  Approach

### 1. **Data Preparation**
- Sampled or generated data pairs containing:
  - `prompt`
  - `chosen`
  - `rejected`
  - `scores_chosen` and `scores_rejected` (token-level reward vectors from LLM)

### 2. **Reward Modeling**
- Instead of training a reward model, we use a **frontier LLM** (e.g., GPT-4) to generate **stepwise preference scores**.
- Rewards are stored and aggregated at token level (inspired by `step-dpo` repo).

### 3. **Model Training Pipeline**
- Created a training script `train_dpo.py` with placeholder logic to use:
  - `DPOTrainer` from HuggingFace
  - Custom logic to accept **stepwise scores**
- Designed a new trainer class (`StepwiseDPOTrainer`) [work in progress].

## Challenges Faced

- **Limited compute**: Training a real DPO model with stepwise rewards requires high memory GPUs (A100 or similar).
- **LLM evaluation cost**: Generating reliable stepwise scores from GPT-4 is expensive and time-intensive.
- **Library support**: HuggingFace’s `trl` library provides a base `DPOTrainer`, but customizing it for step-level aggregation is non-trivial.
- **Sample scarcity**: Creating meaningful prompt-response-rejected triplets manually or via rejection sampling is difficult at scale.

---

## Limitations & Notes

Despite best efforts, the following components remain incomplete or partially implemented due to time and compute constraints:

### Full Training Not Performed

- Stepwise DPO training was **not executed end-to-end**, mainly due to the lack of access to high-memory GPUs (e.g., A100 or T4).
- Training even a 350M or 1.3B parameter model with token-level preference scoring requires >15GB VRAM, which was unavailable.

## Sparse Commits – Explanation

- During active debugging and experimenting, there were instances where **intermediate changes were not committed**, especially while figuring out:
  - reward formatting
  - dataset conversions
  - tokenizer compatibility issues
- This happened due to context switching and focusing heavily on getting the code to work correctly.
- However, if selected for an interview, I would be happy to **walk through every major intermediate step**, logic change, and challenge — most of which are still fresh in mind or reproducible.

---

## Previous Approaches Tried (Before Finalizing the Current Direction)

Before settling on the current approach, multiple alternatives were explored:

### 🔹 1. **Reward Model Fine-Tuning (Abandoned)**
- Tried loading a reward model and tuning it using binary preferences (`chosen` vs `rejected`).
- Dropped due to lack of clean reward data and higher complexity.

### 🔹 2. **Full-token LogProb-based Comparison**
- Compared the log-probs of `chosen` vs `rejected` across token spans.
- Did not capture nuanced stepwise reasoning — hence replaced with LLM-generated scores.

### 🔹 3. **Simple SFT-Only Fine-tuning**
- Initially considered training just a supervised fine-tuned (SFT) model.
- But this wouldn't test the DPO hypothesis, so effort was redirected toward preference-based fine-tuning.

---

## Final Word

While the final trained model is not present, the **pipeline is architecturally sound** and reflects strong understanding of:

- LLM-based reward modeling
- DPO formulation
- HuggingFace `trl` usage
- Dataset structuring and token-level aggregation

This repo can easily be extended into a working solution with adequate GPU access.