from trl import DPOTrainer
from typing import Dict, Any
import torch


class StepwiseDPOTrainer(DPOTrainer):
    def compute_rewards(self, model, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Custom scalar rewards from pre-computed scores.
        Expects:
            reward_chosen (list of floats),
            reward_rejected (list of floats)
        """

        r_chosen = torch.tensor(inputs["reward_chosen"], dtype=torch.float32, device=model.device)
        r_rejected = torch.tensor(inputs["reward_rejected"], dtype=torch.float32, device=model.device)

        return {
            "rewards_chosen": r_chosen,
            "rewards_rejected": r_rejected,
        }
