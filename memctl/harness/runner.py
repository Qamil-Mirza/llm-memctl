"""Build an experiment from a config and run its episodes."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

from memctl.agents import build_agent
from memctl.agents.base import NullAgent
from memctl.analysis.summarize import summarize_episodes
from memctl.config import budget_label, config_hash, resolve
from memctl.controllers import build_controller
from memctl.controllers.base import MemoryController
from memctl.controllers.heuristics import NoController
from memctl.embed import build_embedder
from memctl.envs import build_env
from memctl.harness.episode import UNLIMITED, EpisodeResult, EpisodeSettings, run_episode
from memctl.hindsight.collect import Hindsight, hindsight_from_reference
from memctl.memory.compress import build_compressor, build_consolidator
from memctl.memory.engine import MemoryEngine
from memctl.rewards import RewardFunction
from memctl.runlog import RunLogger, read_jsonl
from memctl.sysinfo import git_state


class Experiment:
    """The components of one experiment, built once and reused for every episode."""

    def __init__(self, config: dict, controller: MemoryController | None = None) -> None:
        self.config = resolve(config)
        memory = self.config["memory"]
        self.seed = int(self.config["seed"])
        self.embedder = build_embedder(memory["embedder"])
        llm = None
        if memory.get("compression_model"):
            from memctl.llm import build_llm

            llm = build_llm(memory["compression_model"])
        self.compressor = build_compressor(memory["compressor"], llm)
        self.consolidator = build_consolidator(memory["consolidator"], self.compressor, llm)
        self.env = build_env(self.config["env"])
        self.agent = build_agent(self.config["agent"])
        self.controller = controller or build_controller(self.config["controller"], self.seed)
        self.shadows = [build_controller(c, self.seed) for c in self.config["shadow_controllers"]]
        self.reward_fn = RewardFunction(self.config["reward"]["weights"])
        self._hindsight: dict[int, Hindsight] = {}

    def engine(self) -> MemoryEngine:
        memory = self.config["memory"]
        return MemoryEngine(memory["allowed_operations"], self.compressor, self.consolidator, memory["compact_ratio"])

    def models(self) -> dict:
        return {
            "task_model": self.agent.model_id,
            "controller_model": self.controller.model_id,
            "embedder": getattr(self.embedder, "name", None),
            "compressor": self.compressor.name,
            "consolidator": self.consolidator.name,
        }

    def hindsight(self, seed: int) -> Hindsight:
        """The reference pass for this seed: unlimited budget, no controller. Cached."""
        if seed not in self._hindsight:
            env = build_env(self.config["env"])
            real = self.agent.cheap or env.stream_depends_on_agent
            agent = build_agent(self.config["agent"]) if real else NullAgent({})
            settings = EpisodeSettings(
                f"reference-{seed}", seed, UNLIMITED, detail=False,
                count_labels=bool(self.config["memory"].get("count_labels", False)),
                store_agent_actions=self.config["memory"]["store_agent_actions"],
                max_steps=self.config["compute_budget"]["max_steps_per_episode"],
            )
            reference = run_episode(env, agent, NoController({}), self.engine(), settings, embedder=None)
            self._hindsight[seed] = hindsight_from_reference(
                reference.state, reference.dependencies, exact=not env.stream_depends_on_agent,
                success=reference.episode["task_success"],
            )
        return self._hindsight[seed]

    def budget_for(self, seed: int) -> tuple[int, int | None, float | None]:
        """(budget in tokens, history tokens, fraction) for one episode."""
        budget = self.config["memory"]["budget"]
        if self.controller.ignores_budget:
            return UNLIMITED, None, None
        if "tokens" in budget:
            return int(budget["tokens"]), None, None
        history = self.hindsight(seed).total_tokens
        return max(1, int(float(budget["fraction"]) * history)), history, float(budget["fraction"])

    def run_episode(
        self, index: int = 0, seed: int | None = None, detail: bool = True, experiment_id: str = "",
        interventions: dict | None = None,
    ) -> EpisodeResult:
        seed = self.seed + index if seed is None else seed
        budget, history, fraction = self.budget_for(seed)
        use_hindsight = self.config["hindsight"]["enabled"] or self.controller.uses_hindsight
        settings = EpisodeSettings(
            episode_id=f"ep{index:05d}",
            seed=seed,
            budget=budget,
            archive_budget=self.config["memory"]["archive_budget"],
            store_agent_actions=self.config["memory"]["store_agent_actions"],
            detail=detail,
            max_steps=self.config["compute_budget"]["max_steps_per_episode"],
            experiment_id=experiment_id,
            budget_fraction=fraction,
            history_tokens=history,
            pricing=self.config["pricing"],
            interventions=interventions or {},
            count_labels=bool(self.config["memory"].get("count_labels", False)),
        )
        return run_episode(
            self.env, self.agent, self.controller, self.engine(), settings, self.reward_fn,
            self.hindsight(seed) if use_hindsight else None, self.embedder, self.shadows,
        )


def run_one(config: dict, seed: int | None = None, controller: MemoryController | None = None, **kwargs) -> EpisodeResult:
    """One episode, nothing written to disk. For tests and analysis."""
    experiment = Experiment(config, controller)
    return experiment.run_episode(seed=seed, **kwargs)


def experiment_id(config: dict) -> str:
    """<date>_<name>_<controller>_<env>_<budget>_seed<seed>_<config hash>_<git>."""
    git = git_state()
    controller = config["controller"].get("label") or config["controller"]["name"]
    return "_".join(
        [
            date.today().isoformat(), config["name"], controller, config["env"]["name"], budget_label(config),
            f"seed{config['seed']}", config_hash(config), git["short"] + ("-dirty" if git["dirty"] else ""),
        ]
    )


def run_experiment(config: dict, folder: str | Path | None = None, progress: bool = False) -> Path:
    """Run every episode of an experiment into a run folder and return the folder.

    If the folder already holds finished episodes of the same config, they are
    kept and only the missing ones are run.
    """
    config = resolve(config)
    experiment = Experiment(config)
    folder = Path(folder) if folder else Path(config["logging"]["output_dir"]) / experiment_id(config)
    logger = RunLogger(folder, copy.deepcopy(config), experiment.models())
    finished = logger.completed()
    checkpoint = folder / "checkpoints" / "controller"
    if finished and checkpoint.exists():
        experiment.controller.load(checkpoint)
    detail_episodes = int(config["logging"]["detail_episodes"])
    try:
        for index in range(int(config["episodes"])):
            if f"ep{index:05d}" in finished:
                continue
            result = experiment.run_episode(index, detail=index < detail_episodes, experiment_id=folder.name)
            logger.write_episode(result)
            if progress:
                episode = result.episode
                print(
                    f"  {episode['episode_id']}  success {episode['task_success']:.3f}  "
                    f"forced {episode['forced_evictions']}  invalid {episode['invalid_actions']}",
                    flush=True,
                )
    except BaseException:
        logger.finalize({}, status="failed")
        raise
    experiment.controller.save(checkpoint)
    summary = summarize_episodes(read_jsonl(folder / "episodes.jsonl"))
    summary.update(experiment_id=folder.name, name=config["name"], models=experiment.models())
    logger.finalize(summary)
    return folder
