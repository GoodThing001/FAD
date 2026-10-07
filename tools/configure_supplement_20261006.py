from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def config(name, entry, arguments, rows):
    p=ROOT/'configs/experiments'/f'{name}.json'
    p.write_text(json.dumps({'name':name,'entrypoint':entry,'arguments':arguments,'acceptance':{'expected_rows':rows,'expect_exit_code':0,'expect_marker':'DONE'}},ensure_ascii=False,indent=2),encoding='utf-8')
config('supplement_history_development_20261006','scripts/reproduction/supplement_cpu_20261006.py',{'mode':'history','fixed':True,'seeds':30,'seed-start':11001,'workers':4},155)
config('supplement_history_confirmation_20261006','scripts/reproduction/supplement_cpu_20261006.py',{'mode':'history','seeds':100,'seed-start':12001,'workers':4},500)
config('supplement_defect_compute_20261006','scripts/reproduction/ensemble_defect.py',{'n-jobs':4},2000)
config('supplement_defect_development_20261006','scripts/reproduction/supplement_cpu_20261006.py',{'mode':'defect','seeds':30,'seed-start':11001,'workers':4,'defect-csv':'runs/srv_supplement_defect_compute_20261006/metrics.csv'},120)
