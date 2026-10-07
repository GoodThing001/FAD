"""Execution-only partition of a frozen 30-seed experiment."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=json.loads((root/'configs/experiments/supplement_paper_development_20261006.json').read_text())
for i,(start,n) in enumerate([(11001,8),(11009,8),(11017,7),(11024,7)],1):
    config=json.loads(json.dumps(base));config['name']=f'supplement_paper_cohort{i}_20261006'
    config['arguments']['seed-start']=start;config['arguments']['seeds']=n
    config['acceptance']['expected_rows']=n*4
    (root/f'configs/experiments/supplement_paper_cohort{i}_20261006.json').write_text(json.dumps(config,indent=2))
aggregate={'name':'supplement_paper_development_aggregate_20261006','entrypoint':'scripts/reproduction/aggregate_paper_20261006.py','arguments':{'seeds':30,'seed-start':11001,'pretrain-epochs':5,'head-epochs':50,'members':10,'corpus-size':2000},'acceptance':{'expected_rows':120,'expect_marker':'DONE','expect_exit_code':0}}
(root/'configs/experiments/supplement_paper_aggregate_20261006.json').write_text(json.dumps(aggregate,indent=2))
