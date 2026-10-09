"""The StreamMemBench adapter (§25) on a small synthetic fixture: no GPU, no network, stub models."""

import json

import pytest

from memctl.envs import build_env
from memctl.envs.streammembench import (
    SIMULATOR_SYSTEM, StreamMemBenchEnv, UserSimulator, chunk_segment, extract_json, sample_items, shard_days,
    token_overlap_score,
)
from memctl.harness.runner import Experiment, run_one
from memctl.llm import ScriptedLLM


def _segment(number, start, lines):
    observations = [
        {"timestamp": f"stream_index:{start + n}", "source": "dialog" if ":" in text else "description",
         "text": text, "metadata": {"stream_index": start + n}}
        for n, text in enumerate(lines)
    ]
    return {"segment_id": number, "time_range": f"11:{number * 5:02d}:00 - 11:{number * 5 + 5:02d}:00",
            "stream_segment": {"text": "\n".join(lines), "observations": observations}}


def _anchor(evidence_id, statement, support):
    return {
        "evidence_id": evidence_id, "evidence_statement": statement, "evidence_type": "stated", "subject": "Tasha (self)",
        "supporting_observations": support,
        "source_span": {"time_window": "", "clip": "A3_TASHA/DAY1", "raw_indices": []},
        "tasks": {
            "initial_task": {"user_request": "Where should I keep the spare hard drive?",
                             "expected_behavior": "Mentions the blue cabinet in the hallway."},
            "followup_task": {"user_request": "Shure cannot find the hard drive. Where should he look?",
                              "expected_behavior": "Points Shure to the blue cabinet in the hallway."},
        },
    }


@pytest.fixture
def root(tmp_path):
    filler = [f"I walked to the table number {n} and looked around the room for a while." for n in range(12)]
    segments = [
        _segment(0, 0, filler),
        _segment(1, 12, ["Jake: The spare hard drive goes in the blue cabinet in the hallway.", *filler]),
        _segment(2, 25, filler),
        _segment(3, 37, filler),
    ]
    good = [{"timestamp": "stream_index:12", "source": "dialog", "text": segments[1]["stream_segment"]["observations"][0]["text"],
             "metadata": {"stream_index": 12, "quote_hint": "Dialogue | Jake: blue cabinet"}}]
    wrong = [{"timestamp": "stream_index:3", "source": "description", "text": "a line that is not there",
              "metadata": {"stream_index": 3}}]
    items = [
        {"participant": "A3_TASHA", "day": "DAY1", "segment_id": 1, "time_range": "", "stream_segment": segments[1]["stream_segment"],
         "evidence_anchors": [_anchor("drive_in_cabinet", "The spare hard drive is kept in the blue cabinet in the hallway.", good)]},
        {"participant": "A3_TASHA", "day": "DAY1", "segment_id": 2, "time_range": "", "stream_segment": segments[2]["stream_segment"],
         "evidence_anchors": [_anchor("label_missing", "The spare hard drive is kept in the blue cabinet in the hallway.", wrong)]},
    ]
    base = tmp_path / "streammembench_v1_en"
    (base / "segments" / "A3_TASHA").mkdir(parents=True)
    (base / "evidence_tasks" / "A3_TASHA").mkdir(parents=True)
    (base / "segments" / "A3_TASHA" / "DAY1.json").write_text(
        json.dumps({"participant": "A3_TASHA", "day": "DAY1", "num_segments": 4, "segments": segments}))
    (base / "evidence_tasks" / "A3_TASHA" / "DAY1.clean.json").write_text(
        json.dumps({"participant": "A3_TASHA", "day": "DAY1", "num_items": 2, "num_evidence_anchors": 2, "items": items}))
    return str(base)


def _scripted_simulator(affirm_initial: bool):
    """Affirms a follow-up or revised answer that names the cabinet; the initial one only if asked to."""

    def reply(prompt):
        answer = prompt.split("assistant_answer:\n", 1)[1].split("\n\nexpected_behavior:", 1)[0]
        revised_or_follow = "Shure" in prompt.split("user_request:\n", 1)[1].split("\n", 1)[0] or "AGAIN" in answer
        ok = "cabinet" in answer and (affirm_initial or revised_or_follow)
        return json.dumps({"type": "affirm" if ok else "revise", "reason": "r",
                           "feedback": "It is in the blue cabinet in the hallway, remember?"})

    return ScriptedLLM(reply, "sim")


def _walk(env, answer):
    observation, kinds = env.reset(0), []
    while observation is not None:
        kinds.append(observation.id.rsplit("#", 1)[1] if "#" in observation.id else "stream")
        result = env.step(answer(observation) if observation.requires_response else None)
        observation = result.observation
    return kinds


def test_chunks_never_cross_a_segment_and_keep_every_line():
    lines = [{"text": f"word {n} " * 10, "metadata": {"stream_index": n}} for n in range(30)]
    chunks = chunk_segment(lines, 50)
    assert [o for c in chunks for o in c] == lines
    assert all(len(c) >= 2 for c in chunks)


def test_vendored_helpers():
    assert token_overlap_score("blue cabinet", "the Blue cabinet") == 1.0
    assert extract_json('noise ```json\n{"type": "affirm"}\n``` more') == {"type": "affirm"}
    assert extract_json("not json") is None
    assert SIMULATOR_SYSTEM.startswith("You are simulating the real user.")


def test_revise_path_then_record_then_followup(root):
    env = StreamMemBenchEnv({"path": root, "sample": {"seed": 0}, "chunk_tokens": 40,
                             "simulator": {"style": "llm", "backend": "stub"}})
    env.simulator = UserSimulator({"style": "llm"}, llm=_scripted_simulator(affirm_initial=False))
    kinds = _walk(env, lambda o: "AGAIN: the blue cabinet" if "#revise" in o.id else "In the blue cabinet.")
    # The stream stops after the last evaluated segment (2); each anchor: init, revise, record, follow.
    assert kinds.count("init") == kinds.count("revise") == kinds.count("record") == kinds.count("follow") == 2
    first = kinds.index("init")
    assert kinds[first:first + 4] == ["init", "revise", "record", "follow"]
    assert "s003" not in " ".join(o.id for o in env._queue)
    stats = env.episode_stats()
    assert stats["anchors"] == 2 and stats["initial_evidence_use"] == 0.0
    assert stats["feedback_incorporation"] == 1.0 and stats["feedback_incorporation_applicable"] == 2
    assert stats["followup_reuse"] == 1.0 == env.task_success()
    assert stats["simulator_calls"] == 6 and stats["simulator_parse_failures"] == 0


def test_affirm_path_skips_the_revision_and_the_record_quotes_the_answer(root):
    env = StreamMemBenchEnv({"path": root, "chunk_tokens": 40})
    env.simulator = UserSimulator({"style": "llm"}, llm=_scripted_simulator(affirm_initial=True))
    env.reset(0)
    seen = []
    observation = env.get_observation()
    while observation is not None:
        seen.append(observation)
        observation = env.step("In the blue cabinet." if observation.requires_response else None).observation
    kinds = [o.id.rsplit("#", 1)[1] for o in seen if "#" in o.id]
    assert kinds == ["init", "record", "follow"] * 2
    record = next(o for o in seen if o.id.endswith("#record"))
    assert record.content.startswith("Tasha says to AI:") and "AI Answer:\nIn the blue cabinet." in record.content
    stats = env.episode_stats()
    assert stats["initial_evidence_use"] == 1.0 and stats["feedback_incorporation"] is None


def test_unparsable_simulator_output_counts_as_revision(root):
    env = StreamMemBenchEnv({"path": root, "chunk_tokens": 40})
    env.simulator = UserSimulator({"style": "llm"}, llm=ScriptedLLM(lambda p: "no json here"))
    _walk(env, lambda o: "something")
    stats = env.episode_stats()
    assert stats["simulator_parse_failures"] == 6 and stats["followup_reuse"] == 0.0


def test_evidence_labels_and_dependencies(root):
    env = StreamMemBenchEnv({"path": root, "chunk_tokens": 40})
    env.reset(0)
    labelled = env._evidence[(1, "drive_in_cabinet")]
    assert labelled == ("A3_TASHA/DAY1/s001/c00",)  # the chunk holding the matched supporting line
    fallback = env._evidence[(2, "label_missing")]
    assert len(fallback) > 1 and all("/s002/" in i for i in fallback)  # no match: the whole segment
    _walk(env, lambda o: "x")
    deps = {d.query_id: d for d in env.get_ground_truth_dependencies()}
    follow = next(d for q, d in deps.items() if q.endswith("drive_in_cabinet#follow"))
    assert follow.requirements[0].item_ids[-1].endswith("drive_in_cabinet#record")


def test_history_tokens_excludes_agent_turns_and_feeds_the_budget(root):
    config = {
        "episodes": 1, "seed": 0,
        "env": {"name": "streammembench", "path": root, "chunk_tokens": 40, "simulator": {"style": "lexical"}},
        "agent": {"name": "llm", "max_new_tokens": 64, "model": {"backend": "stub"}},
        "controller": {"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 2, "method": "lexical", "fit": True},
                       "target_tokens": 60},
        "memory": {"budget": {"fraction": 0.3}, "count_labels": True,
                   "allowed_operations": ["KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]},
        "hindsight": {"enabled": False},
    }
    experiment = Experiment(config)
    experiment.hindsight = lambda seed: pytest.fail("no reference pass may run for this environment")
    budget, history, fraction = experiment.budget_for(0)
    assert history == experiment.env.history_tokens(0, True) and budget == int(0.3 * history)
    result = experiment.run_episode(0)
    episode = result.episode
    assert episode["env"] == "streammembench" and episode["queries"] == episode["env_stats"]["simulator_calls"]
    assert episode["env_stats"]["anchors"] == 2


def test_registered_and_sampled_reproducibly(root):
    env = build_env({"name": "streammembench", "path": root, "sample": {"seed": 0, "items": 1}})
    again = build_env({"name": "streammembench", "path": root, "sample": {"seed": 0, "items": 1}})
    assert env.items == again.items and len(env.items) == 1
    assert build_env({"name": "streammembench", "path": root, "participants": ["A1_JAKE"]}).days == []


def test_shards_cover_every_day_once(root):
    shards = shard_days(root, sample_items(root, ("A3_TASHA",), 0, None), 3)
    assert sorted(d for s in shards for d in s) == [("A3_TASHA", "DAY1")]
    assert build_env({"name": "streammembench", "path": root, "shard": {"k": 3, "index": 1}}).days == shards[1] == []
    excluded = build_env({"name": "streammembench", "path": root, "exclude_days": [["A3_TASHA", "DAY1"]]})
    assert excluded.days == [] and len(excluded.items) == 2  # the sample itself is unchanged


def test_run_one_with_the_rl_free_controllers(root):
    config = {
        "episodes": 1,
        "env": {"name": "streammembench", "path": root, "chunk_tokens": 40, "simulator": {"style": "lexical"}},
        "agent": {"name": "llm", "model": {"backend": "stub"}},
        "controller": {"name": "fifo"},
        "memory": {"budget": {"fraction": 0.5}},
        "hindsight": {"enabled": False},
    }
    result = run_one(config)
    assert result.episode["env_stats"]["anchors"] == 2
    assert result.episode["evidence_complete_rate"] is not None
