# §53 N-dev B0 on the laptop (resumable per row): runs/exp53/dev/b0/N/<case>.json via exp51_stage4.run_b0.
import sys, json
sys.path.insert(0, '/home/qamil-mirza/Code/llm-memctl'); sys.path.insert(0, '/home/qamil-mirza/Code/llm-memctl/scripts')
import exp51_engine as EN, exp51_common as CM, exp51_stage4 as S4
from pathlib import Path
rows = [json.loads(l) for l in open('/home/qamil-mirza/Code/llm-memctl/runs/exp53/cases53.jsonl')]
rows = [r for r in rows if r['nsplit'] == 'dev']
e = EN.load('hf', 16)
S4.run_b0(e, rows, Path('/home/qamil-mirza/Code/llm-memctl/runs/exp53/dev'), 'N', CM.t_cities())
print('done', len(rows))
