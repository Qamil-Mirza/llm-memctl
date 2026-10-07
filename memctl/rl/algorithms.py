"""Learning algorithms for the item policy, behind one interface.

An algorithm turns a batch of recorded decisions into one parameter update and
returns statistics. To add one (DQN-style, a contextual bandit, GRPO, ...),
write a class with `update` and add it to `ALGORITHMS`; the controller, the
harness and the trainer loop do not change.

What each algorithm needs on a `Decision`:
- `bc`         expert labels (`expert_rank`, `expert_retrieved`)
- `cost`       expert costs (`expert_cost`), plus `expert_rank` and `expert_retrieved`
- `reinforce`  `return_to_go`
- `ppo`        `return_to_go`, and the `log_prob` and `value` recorded when it was taken
- `grpo`       `advantage` (set by the trainer from a group of samples of one episode) and `log_prob`;
               expert labels too when `imitation_weight` is set
"""

from __future__ import annotations

import random

import numpy as np
import torch

from memctl.controllers.rl import N_REMOVALS, Decision, accepted_picks, evaluate
from memctl.rl.policy import RETRIEVE_COLUMN, ItemPolicy


def imitation_loss(policy: ItemPolicy, decision: Decision) -> tuple[torch.Tensor, int, int]:
    """(loss, picks that matched the expert, picks) for one decision.

    The expert's removal sequence is followed step by step. At each step every
    pair the expert would accept counts as correct, so the loss is the negative
    log of the probability mass the policy puts on that set.
    """
    logits, _ = policy(torch.from_numpy(decision.items), torch.from_numpy(decision.global_features))
    n = decision.n_active
    loss = logits.new_zeros(())
    matched = total = 0
    excess = decision.excess
    if decision.expert_retrieved:
        target = torch.tensor(decision.expert_retrieved, dtype=torch.float32)
        retrieve_logits = logits[n:, RETRIEVE_COLUMN]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(retrieve_logits, target, reduction="sum")
        matched += int(((retrieve_logits > 0).float() == target).sum())
        total += len(decision.expert_retrieved)
        excess += sum(t for t, flag in zip(decision.retrieve_tokens, decision.expert_retrieved) if flag)
    mask = torch.from_numpy(decision.mask).clone()
    while excess > 0:
        accepted = accepted_picks(decision.expert_rank, mask)
        if not accepted:
            break
        log_probs = torch.log_softmax(logits[:n, :N_REMOVALS].masked_fill(~mask, -1e9).flatten(), dim=0)
        loss = loss - torch.logsumexp(log_probs[accepted], dim=0)
        matched += int(int(torch.argmax(log_probs)) in accepted)
        total += 1
        pick = accepted[0]
        mask[pick // N_REMOVALS] = False
        excess -= int(decision.savings[pick // N_REMOVALS, pick % N_REMOVALS])
    return loss, matched, total


def cost_loss(policy: ItemPolicy, decision: Decision) -> tuple[torch.Tensor, int, int]:
    """(loss, picks of least cost, picks) for one decision: cost-sensitive imitation.

    Along the expert's removal sequence, the loss at each pick is the policy's
    expected cost, sum over available pairs of probability times cost. Its
    minimiser, for a learner that sees only part of the state, is the pair of
    least *expected* cost given what it sees (AggreVaTe's cost-sensitive
    classification), not the pair most often of least cost. Retrieval is
    learned as in `imitation_loss`.
    """
    logits, _ = policy(torch.from_numpy(decision.items), torch.from_numpy(decision.global_features))
    n = decision.n_active
    loss = logits.new_zeros(())
    matched = total = 0
    excess = decision.excess
    if decision.expert_retrieved:
        target = torch.tensor(decision.expert_retrieved, dtype=torch.float32)
        retrieve_logits = logits[n:, RETRIEVE_COLUMN]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(retrieve_logits, target, reduction="sum")
        matched += int(((retrieve_logits > 0).float() == target).sum())
        total += len(decision.expert_retrieved)
        excess += sum(t for t, flag in zip(decision.retrieve_tokens, decision.expert_retrieved) if flag)
    costs = torch.from_numpy(np.nan_to_num(decision.expert_cost, nan=0.0)).flatten()
    mask = torch.from_numpy(decision.mask & ~np.isnan(decision.expert_cost)).clone()
    while excess > 0:
        accepted = accepted_picks(decision.expert_rank, mask)
        if not accepted:
            break
        probs = torch.softmax(logits[:n, :N_REMOVALS].masked_fill(~mask, -1e9).flatten(), dim=0)
        loss = loss + (probs * costs).sum()
        matched += int(int(torch.argmax(probs)) in accepted)
        total += 1
        pick = accepted[0]
        mask[pick // N_REMOVALS] = False
        excess -= int(decision.savings[pick // N_REMOVALS, pick % N_REMOVALS])
    return loss, matched, total


class BehaviourCloning:
    name = "bc"
    loss = staticmethod(imitation_loss)

    def __init__(self, settings: dict) -> None:
        self.epochs = int(settings.get("epochs", 4))
        self.batch_size = int(settings.get("batch_size", 64))

    def update(self, policy: ItemPolicy, optimizer: torch.optim.Optimizer, decisions: list[Decision], rng: random.Random) -> dict:
        labelled = [d for d in decisions if d.expert_rank is not None]
        if self.name == "cost":
            labelled = [d for d in labelled if d.expert_cost is not None]
        total_loss = matched = picks = 0.0
        for _ in range(self.epochs):
            order = labelled[:]
            rng.shuffle(order)
            total_loss = matched = picks = 0.0
            for start in range(0, len(order), self.batch_size):
                losses = []
                for decision in order[start : start + self.batch_size]:
                    loss, hit, count = self.loss(policy, decision)
                    if count:
                        losses.append(loss)
                        matched += hit
                        picks += count
                if not losses:
                    continue
                batch_loss = torch.stack(losses).sum() / max(1.0, len(losses))
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                total_loss += batch_loss.item() * len(losses)
        return {
            "loss": total_loss / max(1, len(labelled)),
            "expert_agreement": matched / picks if picks else None,
            "decisions": len(labelled),
        }


class CostSensitiveImitation(BehaviourCloning):
    name = "cost"
    loss = staticmethod(cost_loss)


class PolicyGradient:
    """REINFORCE with a learned baseline, or PPO when `clip` is set and `epochs` > 1."""

    name = "ppo"

    def __init__(self, settings: dict) -> None:
        self.epochs = int(settings.get("epochs", 4))
        self.clip = settings.get("clip", 0.2)
        self.value_coef = float(settings.get("value_coef", 0.5))
        self.entropy_coef = float(settings.get("entropy", 0.01))
        self.batch_size = int(settings.get("batch_size", 256))

    def update(self, policy: ItemPolicy, optimizer: torch.optim.Optimizer, decisions: list[Decision], rng: random.Random) -> dict:
        decisions = [d for d in decisions if d.picks or d.retrieved]
        if not decisions:
            return {"loss": None, "decisions": 0}
        returns = torch.tensor([d.return_to_go for d in decisions], dtype=torch.float32)
        advantages = returns - torch.tensor([d.value for d in decisions], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        old_log_probs = torch.tensor([d.log_prob for d in decisions], dtype=torch.float32)
        last = {}
        for _ in range(self.epochs):
            order = list(range(len(decisions)))
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                policy_terms, value_terms, entropies = [], [], []
                for index in batch:
                    log_prob, value, entropy = evaluate(policy, decisions[index])
                    ratio = torch.exp(log_prob - old_log_probs[index])
                    if self.clip is None:
                        policy_terms.append(-log_prob * advantages[index])
                    else:
                        clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                        policy_terms.append(-torch.min(ratio * advantages[index], clipped * advantages[index]))
                    value_terms.append((value - returns[index]) ** 2)
                    entropies.append(entropy)
                policy_loss = torch.stack(policy_terms).mean()
                value_loss = torch.stack(value_terms).mean()
                entropy = torch.stack(entropies).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                last = {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                        "entropy": entropy.item()}
        return {**last, "decisions": len(decisions), "mean_return": float(returns.mean())}


class GroupRelative(PolicyGradient):
    """GRPO: PPO's clipped objective with no value network. The trainer runs each episode
    `group` times with different action samples and sets every decision's `advantage` to its
    episode's return minus the group mean, over the group's standard deviation. Same seed, same
    budget, so what is left of the spread is the policy's own choices, not the episode's luck."""

    name = "grpo"

    def __init__(self, settings: dict) -> None:
        super().__init__(settings)
        # Optional: keep the expert's imitation loss in the objective (CHORD / SRFT style), on a
        # sample of labelled decisions as large as each policy-gradient batch.
        self.imitation_weight = float(settings.get("imitation_weight", 0.0))

    def update(self, policy: ItemPolicy, optimizer: torch.optim.Optimizer, decisions: list[Decision], rng: random.Random) -> dict:
        labelled = [d for d in decisions if d.expert_rank is not None] if self.imitation_weight else []
        decisions = [d for d in decisions if (d.picks or d.retrieved) and d.advantage]
        if not decisions:
            return {"loss": None, "decisions": 0}
        advantages = torch.tensor([d.advantage for d in decisions], dtype=torch.float32)
        old_log_probs = torch.tensor([d.log_prob for d in decisions], dtype=torch.float32)
        last = {}
        for _ in range(self.epochs):
            order = list(range(len(decisions)))
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                policy_terms, entropies = [], []
                for index in batch:
                    log_prob, _, entropy = evaluate(policy, decisions[index])
                    ratio = torch.exp(log_prob - old_log_probs[index])
                    clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                    policy_terms.append(-torch.min(ratio * advantages[index], clipped * advantages[index]))
                    entropies.append(entropy)
                policy_loss = torch.stack(policy_terms).mean()
                entropy = torch.stack(entropies).mean()
                loss = policy_loss - self.entropy_coef * entropy
                if labelled:
                    terms = [imitation_loss(policy, d)[0] for d in rng.sample(labelled, min(len(labelled), len(batch)))]
                    imitation = torch.stack(terms).mean()
                    loss = loss + self.imitation_weight * imitation
                    last["imitation_loss"] = imitation.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                last.update({"loss": loss.item(), "policy_loss": policy_loss.item(), "entropy": entropy.item()})
        return {**last, "decisions": len(decisions)}


class Reinforce(PolicyGradient):
    name = "reinforce"

    def __init__(self, settings: dict) -> None:
        super().__init__({**settings, "epochs": 1, "clip": None})


ALGORITHMS = {"bc": BehaviourCloning, "cost": CostSensitiveImitation, "grpo": GroupRelative, "ppo": PolicyGradient, "reinforce": Reinforce}
