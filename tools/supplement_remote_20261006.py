"""Narrow remote inventory/launch/download for authorized supplemental jobs."""
import sys, json, shlex, os, platform, shutil
from pathlib import Path
from sync_remote import connect, push, _mkdirs, REMOTE, ROOT, run

def main():
    c=connect()
    try:
        mode=sys.argv[1]
        if mode=='inventory':
            code="import os,shutil,json,pathlib,importlib.util; print(json.dumps({'cpu':os.cpu_count(),'load':os.getloadavg(),'disk_free':shutil.disk_usage('/home/hzeng').free,'envs':list(str(p) for p in pathlib.Path('/home/hzeng/envs').glob('*/bin/python*')),'corpus_exists':pathlib.Path('/home/hzeng/project/FAD_CLEAN/data/ngs_sequences.txt').exists()},indent=2))"
            print(run(c,'/usr/bin/python3 -c '+shlex.quote(code)))
            print(run(c,"find /home/hzeng/project/FAD_CLEAN -maxdepth 3 -iname '*dock*'"))
        elif mode=='push':push(c,sys.argv[2:])
        elif mode=='preflight':
            code="import json,hashlib,pathlib,importlib.util; root=pathlib.Path('/home/hzeng/project/FAD_CLEAN'); print(json.dumps({'python':__import__('sys').executable,'data_sha256':hashlib.sha256((root/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').read_bytes()).hexdigest(),'packages':{k:bool(importlib.util.find_spec(k)) for k in ['numpy','pandas','scipy','sklearn','torch','nupack']}})); import torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'arch':torch.cuda.get_arch_list()}))" if sys.argv[3]=='torch' else "import nupack,numpy,pandas,scipy,json,hashlib,pathlib; root=pathlib.Path('/home/hzeng/project/FAD_CLEAN'); print(json.dumps({'nupack':nupack.__version__,'data_sha256':hashlib.sha256((root/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').read_bytes()).hexdigest()}))"
            print(run(c,shlex.quote(sys.argv[2])+' -c '+shlex.quote(code),timeout=45))
        elif mode=='launch':
            env,config,rid=sys.argv[2:5];gpu=sys.argv[5] if len(sys.argv)>5 else '0'
            # Close every inherited SSH descriptor; otherwise background shell
            # channels remain open and the launcher times out despite a live job.
            code="import subprocess,pathlib,json; root=pathlib.Path("+repr(REMOTE)+"); rid="+repr(rid)+"; assert not (root/'runs'/rid).exists(), 'run already exists'; log=open(root/'runs'/(rid+'_launcher.log'),'x'); p=subprocess.Popen("+repr(['nohup','nice','-n','15','env','PYTHONUNBUFFERED=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',f'CUDA_VISIBLE_DEVICES={gpu}',env,'scripts/run_experiment.py',config,'--run-id',rid])+",cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True); print(json.dumps({'run_id':rid,'pid':p.pid}))"
            cmd='/usr/bin/python3 -c '+shlex.quote(code)
            print(run(c,cmd,timeout=20))
            print('launched',rid)
        elif mode=='status':
            for rid in sys.argv[2:]:
                print(rid,run(c,'tail -n 5 '+shlex.quote(REMOTE+'/runs/'+rid+'/stdout.log'),timeout=20))
        elif mode=='status-json':
            code="import pathlib,json,csv; root=pathlib.Path("+repr(REMOTE)+")/'runs'; result=[]\nfor rid in "+repr(sys.argv[2:])+":\n p=root/rid; f=p/'metrics.csv'; rows=list(csv.DictReader(f.open())) if f.exists() else []; result.append({'run_id':rid,'DONE':(p/'DONE').exists(),'FAILED':(p/'FAILED').exists(),'rows':len(rows),'completed_seeds':len({r['seed'] for r in rows})})\nprint(json.dumps(result))"
            print(run(c,'/usr/bin/python3 -c '+shlex.quote(code),timeout=20))
        elif mode=='stop-monolithic-paper':
            rid=sys.argv[2];assert rid=='srv_supplement_paper_dev_20261006'
            code="import os,signal,pathlib,json; target="+repr('runs/'+rid+'/metrics.csv')+"; stopped=[]\nfor d in pathlib.Path('/proc').iterdir():\n if not d.name.isdigit(): continue\n try: args=(d/'cmdline').read_bytes().decode().split('\\0')\n except (FileNotFoundError,PermissionError,UnicodeDecodeError): continue\n if target in args and any(a.endswith('/supplement_paper_20261006.py') for a in args):\n  os.kill(int(d.name),signal.SIGTERM); stopped.append(int(d.name))\nprint(json.dumps({'stopped_exact_experiment_pids':stopped,'reason':'execution-only cohort scheduling; retain partial run as FAILED'}))"
            print(run(c,'/usr/bin/python3 -c '+shlex.quote(code),timeout=20))
        elif mode=='pull':
            rid=sys.argv[2]; dest=ROOT/'evidence/supplement_20261006'/rid;sftp=c.open_sftp()
            def rec(remote,local):
                local.mkdir(parents=True,exist_ok=True)
                import stat
                for a in sftp.listdir_attr(remote):
                    if stat.S_ISDIR(a.st_mode):rec(remote+'/'+a.filename,local/a.filename)
                    elif a.filename.endswith(('.csv','.json','.log','.jsonl')) or a.filename in ['DONE','FAILED']:sftp.get(remote+'/'+a.filename,str(local/a.filename))
            rec(REMOTE+'/runs/'+rid,dest);sftp.close();print('downloaded',rid)
    finally:c.close()
if __name__=='__main__':main()
