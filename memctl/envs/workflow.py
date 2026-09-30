"""Phase 2: a sequential tool-use task where forgetting changes what happens next.

Several jobs run interleaved. Each job has stages that must be run in order, and
running a stage needs the token that the previous stage of the same job
returned. Between two stages of one job come the other jobs' stages and log
noise, so the token has to survive in memory for a long time.

Unlike the recall task, a memory failure here has consequences: an agent that
no longer has the token must restart the job from its first stage, which costs
steps from a fixed step budget, produces new tokens and new observations, and
can push other jobs' tokens out in turn. The observation stream therefore
depends on the memory policy (`stream_depends_on_agent`), so hindsight from the
reference pass is exact only for a policy that never forgets what it needs.

Task success is the share of jobs completed when the step budget runs out.
"""

from __future__ import annotations

import random
import re

from memctl.envs.base import TaskEnvironment
from memctl.envs.grammar import fact_sentence
from memctl.envs.synthetic import FILLER
from memctl.memory.items import SourceType
from memctl.task import Dependency, EvidenceRequirement, Observation, StepResult
from memctl.util import merged

DEFAULTS = {
    "horizon": 400,  # the step budget
    "jobs": 16,
    "stages": 4,
    "noise_prob": 0.5,  # share of free steps that are log noise instead of an instruction
    "long_noise_prob": 0.3,
}
STAGES = ["intake", "review", "approval", "dispatch", "closing", "audit"]
ACTION = re.compile(r"(run|restart)\s+(?:(\w+)\s+)?(job-\d+)(?:\s+([\w-]+))?")
INSTRUCTION = re.compile(r"Instruction: run the (\w+) stage of (job-\d+)(?: using its (\w+) token)?\.")


def instruction_sentence(stage: str, job: str, previous: str | None) -> str:
    using = f" using its {previous} token" if previous else ""
    return f"Instruction: run the {stage} stage of {job}{using}."


class WorkflowEnv(TaskEnvironment):
    name = "workflow"
    goal = (
        "Run each job's stages in order. A stage needs the token returned by the job's previous stage. "
        "Reply 'run <stage> <job> <token>', or 'restart <job>' if you no longer have the token."
    )
    stream_depends_on_agent = True

    def __init__(self, config: dict) -> None:
        super().__init__(merged(DEFAULTS, config))
        self.horizon = int(self.config["horizon"])
        self.stages = STAGES[: int(self.config["stages"])]
        if len(self.stages) < 2:
            raise ValueError("workflow needs at least 2 stages")
        self._rng = random.Random(0)
        self._step = 0
        self._current: Observation | None = None

    # ---- TaskEnvironment ---------------------------------------------------

    def reset(self, seed: int) -> Observation:
        self._rng = random.Random(seed)
        self._step = 0
        jobs = [f"job-{100 + n}" for n in range(int(self.config["jobs"]))]
        self._progress = {job: 0 for job in jobs}  # the stage the job is waiting to run
        self._token: dict[str, tuple[str, str, str]] = {}  # job -> (token, id of the item stating it, sentence)
        self._pending: tuple[str, int] | None = None  # (job, stage) whose result is shown next
        self._asked: tuple[str, int] | None = None  # the instruction the agent is answering
        self._dependencies: dict[str, Dependency] = {}
        self._last_reward = 0.0
        self._stats = {"instructions": 0, "succeeded": 0, "restarts": 0, "wrong_tokens": 0, "completed_jobs": 0}
        self._current = self._next_observation()
        return self._current

    def step(self, agent_action: str | None) -> StepResult:
        info: dict = {"scored": False, "correct": None}
        self._last_reward = 0.0
        if self._asked is not None:
            job, stage = self._asked
            ok = self._execute(job, stage, agent_action or "")
            self._stats["instructions"] += 1
            self._stats["succeeded"] += int(ok)
            info = {"scored": True, "correct": ok, "query_id": self._current.id,
                    "gold": self._token.get(job, ("",))[0] if stage > 0 and not ok else None}
            self._asked = None
        done = self._step >= self.horizon or self._stats["completed_jobs"] == len(self._progress)
        self._current = None if done else self._next_observation()
        return StepResult(self._current, self._last_reward, self._current is None, info)

    def get_observation(self) -> Observation | None:
        return self._current

    def is_done(self) -> bool:
        return self._current is None

    def get_reward(self) -> float:
        return self._last_reward

    def task_success(self) -> float:
        return self._stats["completed_jobs"] / len(self._progress)

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        return list(self._dependencies.values())

    def episode_stats(self) -> dict:
        return dict(self._stats)

    # ---- mechanics ------------------------------------------------------------

    def _execute(self, job: str, stage: int, action: str) -> bool:
        """Apply the agent's reply to the instruction for (job, stage). True if the stage ran."""
        match = ACTION.search(action)
        expected = self._token.get(job, (None,))[0] if stage > 0 else None
        ran = (
            match is not None and match.group(1) == "run" and match.group(3) == job
            and match.group(2) == self.stages[stage] and (stage == 0 or match.group(4) == expected)
        )
        if not ran:  # no token, a wrong token or no sensible reply: the job starts over
            if match is not None and match.group(1) == "run":
                self._stats["wrong_tokens"] += 1
            self._stats["restarts"] += 1
            self._progress[job] = 0
            self._token.pop(job, None)
            return False
        self._pending = (job, stage)
        return True

    def _new_token(self) -> str:
        letters, digits = "ABCDEFGHJKLMNPQRSTUVWXYZ", "0123456789"
        rng = self._rng
        return rng.choice(letters) + rng.choice(digits) + rng.choice(digits) + rng.choice(letters)

    def _next_observation(self) -> Observation:
        self._step += 1
        item_id = f"o{self._step:05d}"
        if self._pending is not None:  # the tool's result comes straight after the action
            job, stage = self._pending
            self._pending = None
            if stage == len(self.stages) - 1:
                self._progress[job] = len(self.stages)
                self._stats["completed_jobs"] += 1
                self._last_reward = 1.0
                return Observation(item_id, f"The {self.stages[stage]} stage finished and {job} is complete.", SourceType.TOOL_OUTPUT)
            token = self._new_token()
            sentence = fact_sentence(self._step, f"{self.stages[stage]} token", job, token)
            self._token[job] = (token, item_id, sentence)
            self._progress[job] = stage + 1
            return Observation(item_id, sentence, SourceType.TOOL_OUTPUT)

        waiting = [job for job, stage in self._progress.items() if stage < len(self.stages)]
        if not waiting or self._rng.random() < self.config["noise_prob"]:
            long = self._rng.random() < self.config["long_noise_prob"]
            count = self._rng.randint(4, 9) if long else self._rng.randint(1, 2)
            text = " ".join(self._rng.choice(FILLER) for _ in range(count))
            return Observation(item_id, text, SourceType.TOOL_OUTPUT if long else SourceType.OBSERVATION)

        job = self._rng.choice(waiting)
        stage = self._progress[job]
        previous = self.stages[stage - 1] if stage > 0 else None
        self._asked = (job, stage)
        requirements = ()
        if stage > 0:
            _, source_id, sentence = self._token[job]
            requirements = (EvidenceRequirement((source_id,), sentence),)
        self._dependencies[item_id] = Dependency(item_id, self._step, requirements, category=f"stage_{stage}")
        text = instruction_sentence(self.stages[stage], job, previous)
        return Observation(item_id, text, SourceType.USER, requires_response=True)
