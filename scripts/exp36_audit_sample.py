"""§36 hand audit sample (folds 0-1 only): 20 B candidates judged correct, 20 judged wrong, 20 F-cut candidates.
Writes the sample with the judge verdict hidden (runs/exp36_audit/sample.jsonl) and the key separately (key.jsonl)."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from exp35_report import load
from memctl.jacobian_data import INPUT_HEADER
from memctl.analysis.jacobian import ablate_evidence
data, qids, _ = load({0, 1}, Path('runs/exp35'), 'runs', Path('runs/exp34_jlens_own'))
d = data['qwen3b']; rng = random.Random(36)
fixed = [q for q in d.frozen if d.correct('B', q) is not None and not d.rows['B'][q]['refusal']]
pos = sorted(q for q in fixed if d.correct('B', q)); neg = sorted(q for q in fixed if not d.correct('B', q))
fcut = sorted(q for q in d.qids if d.correct('Fcut', q) is not None and not d.rows['Fcut'][q]['refusal'])
pick = [('B', q) for q in rng.sample(pos, 20)] + [('B', q) for q in rng.sample(neg, 20)] + [('Fcut', q) for q in rng.sample(fcut, 20)]
rng.shuffle(pick)
out = Path('runs/exp36_audit'); s = open(out / 'sample.jsonl', 'w'); k = open(out / 'key.jsonl', 'w')
for i, (arm, q) in enumerate(pick):
    r = d.by_q[q]
    s.write(json.dumps({'i': i, 'arm': arm, 'question_id': q, 'question': r.prompt.split(INPUT_HEADER, 1)[1].strip()[:400], 'evidence': list(r.evidence_lines),
                        'gold_in_cut_prompt': (r.gold.lower() in ablate_evidence(r.prompt, r.memory_lines, r.evidence_lines).lower()) if arm == 'Fcut' else None, 'gold': r.gold,
                        'type': r.question_type, 'candidate': d.rows[arm][q]['final_answer']}) + '\n')
    k.write(json.dumps({'i': i, 'arm': arm, 'question_id': q, 'judge': d.correct(arm, q)}) + '\n')
