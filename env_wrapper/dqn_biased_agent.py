"""Custom DQN agent variants for notebook-safe multiprocessing imports."""

import torch
from elegantrl.agents import AgentDQN
from elegantrl.train.config import Config


class AgentDQNBiased(AgentDQN):
    """DQN with prior-weighted random action sampling in epsilon exploration."""

    def __init__(self, net_dims, state_dim, action_dim, gpu_id=0, args=Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

        probs = getattr(args, "action_prior_probs", None)
        if probs is None:
            probs = [1.0 / action_dim] * action_dim

        prior = torch.tensor(probs, dtype=torch.float32)
        if prior.numel() != action_dim:
            raise ValueError(f"action_prior_probs length should be {action_dim}, got {prior.numel()}")
        if torch.any(prior < 0):
            raise ValueError("action_prior_probs cannot contain negative values")

        total = float(prior.sum().item())
        if total <= 0:
            raise ValueError("action_prior_probs sum must be positive")

        self.action_prior_probs = (prior / total).to(self.device)

    def explore_action(self, state):
        batch_size = state.shape[0]
        if self.explore_rate < torch.rand(1, device=state.device):
            return self.act.get_q_value(state).argmax(dim=1)

        probs = self.action_prior_probs.unsqueeze(0).expand(batch_size, -1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
