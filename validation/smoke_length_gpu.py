"""Bounded original-model GPU length checks. No formal training launch."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'log/length_gpu_smoke_20261120_v1'
SOURCE=RUN/'source'
PYTHON='/data/xiaozj/anaconda3/envs/exp/bin/python'
DATASETS=('iscx','ssrc','mirage','utmn')


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)


def prepare():
    RUN.mkdir(parents=True,exist_ok=False)
    original=ROOT/'log/binning_formal_20261117_v1/source'
    manifest={'formal_training':False,'batches':[1,2,4,8,16,32,64],'steps_per_batch_size':2,
              'gpu_map':{'0':32,'1':64},'allocator_fraction':.5,'process_gpu_stop_mib':6144,
              'minimum_free_gpu_mib':4096,'worker_timeout_seconds':180,'source_sha256':{},'inputs':{}}
    for folder in ('stateFormer','pre_process','post_process'):
        for p in (original/folder).glob('*.py'):
            target=SOURCE/folder/p.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(p.read_bytes());manifest['source_sha256'][str(target.relative_to(SOURCE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert 'beta = 0.01' in (SOURCE/'stateFormer/model_train.py').read_text()
    assert 'self.noise_heads = nn.ModuleList' in (SOURCE/'stateFormer/transformer.py').read_text()
    for dataset in DATASETS:
        target=RUN/'input'/dataset;target.mkdir(parents=True)
        counts={}
        for p in sorted((ROOT/'data-json'/dataset).glob('*.json')):
            flows=json.loads(p.read_text())
            # At most two long and one short real flows per class; truncate on CPU.
            long=[];short=[]
            for flow in flows:
                if len(flow['series'])>=64 and len(long)<2:long.append(flow)
                if len(flow['series'])<32 and not short:short.append(flow)
                if len(long)==2 and short:break
            selected=long+short
            if not selected:selected=flows[:2]
            # Classes lacking 64-packet flows retain real shorter examples.
            compact=[dict(flow,series=flow['series'][:64]) for flow in selected]
            (target/p.name).write_text(json.dumps(compact));counts[p.stem]=len(compact)
            del flows,long,short,selected,compact
        manifest['inputs'][dataset]=counts
        for length in (32,64):
            src=ROOT/f'log/length_precheck_20261120_v2/length_{length}/bins/bins_{dataset}.json'
            dest=RUN/f'length_{length}'/'bins'/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(src.read_bytes())
    dump(RUN/'manifest.json',manifest)
    print('Prepared bounded real-data fixtures and original source snapshot',flush=True)


def worker(gpu,dataset):
    length={'0':32,'1':64}[gpu]
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    sys.path.insert(0,str(SOURCE))
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from stateFormer.transformer import TransGenerator
    from stateFormer.dataset import CustomDataset
    from stateFormer.model_train import transformer_train
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(.5,0)
    torch.manual_seed(20261120)
    mapping={p.stem:i for i,p in enumerate(sorted((RUN/'input'/dataset).glob('*.json')))}
    fields=['time','pkt_len','flags','ttl'];meta=['src_port','dst_port','src_ip','dst_ip']
    binsfile=RUN/f'length_{length}'/'bins'/f'bins_{dataset}.json'
    bins=json.loads(binsfile.read_text());sizes=[len(v['intervals']) for v in bins.values()]
    data=CustomDataset(str(RUN/'input'/dataset)+'/',str(binsfile),mapping,length,meta,fields,preencode=True)
    seqs,labels,lengths,weights=data._encoded
    # Stress dense full-length batches using actual longest examples. Original
    # tensors remain available for the separate real mixed-length check.
    longest=torch.argsort(lengths,descending=True)[:max(1,int((lengths==length).sum()))]
    if not (lengths[longest]==length).all():
        raise RuntimeError('No full-length real flow available for stress check')
    generator=TransGenerator(len(mapping),8,length,sizes,torch.device('cuda')).cuda()
    report={'gpu':int(gpu),'dataset':dataset,'length':length,'samples':len(data),'classes':len(mapping),
            'total_bins':sum(sizes),'beta':.01,'noise_heads':True,'device':torch.cuda.get_device_name(0),
            'allocator_fraction':.5,'stages':[],'status':'running'}
    path=RUN/f'gpu{gpu}_{dataset}.json';dump(path,report)
    previous_reserved=None
    with tempfile.TemporaryDirectory(prefix='temporary-checkpoint-',dir=RUN) as temporary:
        for batch in (1,2,4,8,16,32,64):
            if previous_reserved and previous_reserved*2>4.5*2**30:
                report.update(status='stopped_at_memory_budget',next_batch=batch);dump(path,report);return
            indices=longest[torch.arange(batch*2)%len(longest)]
            loader=DataLoader(TensorDataset(seqs[indices],labels[indices],lengths[indices],weights[indices]),batch_size=batch,shuffle=False)
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            generator.train()
            transformer_train(generator,loader,1,torch.device('cuda'),temporary+'/',0)
            torch.cuda.synchronize()
            assert all(torch.isfinite(p).all().item() for p in generator.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in generator.parameters())
            stats={'batch':batch,'steps':2,'full_length_real_flows':True,'seconds':time.perf_counter()-start,
                   'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                   'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
            previous_reserved=torch.cuda.max_memory_reserved()
            report['stages'].append(stats);dump(path,report);print('STAGE',stats,flush=True)
        # One mixed-length batch checks padding/mask using unmodified real samples.
        subset=torch.arange(min(len(data),8))
        torch.cuda.reset_peak_memory_stats()
        transformer_train(generator,DataLoader(TensorDataset(seqs[subset],labels[subset],lengths[subset],weights[subset]),batch_size=len(subset)),1,torch.device('cuda'),temporary+'/',0)
        generator.eval()
        with torch.no_grad():
            label=labels[:4].cuda();lens=lengths[:4].cuda()
            samples=generator.sample(len(label),label,lens)
            assert samples.shape==(len(label),length,8)
            for i,count in enumerate(sizes):assert ((samples[:,:,i]>=1)&(samples[:,:,i]<=count)).all().item()
            prefix=samples[:,:2,:]
            continuation=generator.sample(len(label),label,lens,x=prefix)
            assert continuation.shape==samples.shape and torch.equal(continuation[:,:2],prefix)
        torch.cuda.synchronize()
        report['mixed_length_and_sampling_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        report['mixed_length_and_sampling_peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
    report.update(status='passed',verified_batch=64,training_steps=15,sampling_cases=2,formal_training=False)
    dump(path,report)
    print('PASSED',gpu,dataset,length,flush=True)


def gpu_state():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
    return {row.split(',')[0].strip():list(map(int,row.split(',')[1:])) for row in raw.splitlines()}


def run_checks():
    manifest=json.loads((RUN/'manifest.json').read_text())
    for relative,digest in manifest['source_sha256'].items():assert hashlib.sha256((SOURCE/relative).read_bytes()).hexdigest()==digest
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=5)
    assert not apps.strip(),'GPU compute processes present; refusing verification'
    queues={'0':list(DATASETS),'1':list(DATASETS)};active={};finished=[];failure=None;observed={}
    def launch(gpu):
        dataset=queues[gpu].pop(0)
        log=(RUN/f'gpu{gpu}_{dataset}.log').open('x')
        child=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'worker','--gpu',gpu,'--dataset',dataset],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(child,dataset,time.monotonic(),log)
        observed[(gpu,dataset)]={'peak_device_used_mib':0,'minimum_device_free_mib':10240,'peak_temperature_c':0}
        print('START',gpu,dataset,'PID',child.pid,flush=True)
    for gpu in queues:launch(gpu)
    try:
        while active:
            states=gpu_state()
            for gpu,(child,dataset,start,log) in list(active.items()):
                used,free,temp=states[gpu]
                monitor=observed[(gpu,dataset)]
                monitor['peak_device_used_mib']=max(monitor['peak_device_used_mib'],used)
                monitor['minimum_device_free_mib']=min(monitor['minimum_device_free_mib'],free)
                monitor['peak_temperature_c']=max(monitor['peak_temperature_c'],temp)
                if used>6144 or free<4096 or temp>=85 or time.monotonic()-start>180:
                    failure=f'Budget/temperature/timeout stop: gpu={gpu}, used={used}, free={free}, temperature={temp}'
                    raise RuntimeError(failure)
                code=child.poll()
                if code is not None:
                    log.close();del active[gpu]
                    if code!=0:raise RuntimeError(f'Worker failed: GPU{gpu}/{dataset}, exit={code}')
                    result=json.loads((RUN/f'gpu{gpu}_{dataset}.json').read_text())
                    result['external_gpu_monitor']=monitor
                    text=(RUN/f'gpu{gpu}_{dataset}.log').read_text()
                    assert not re.search(r'(?:Loss|NLL|KL): (?:nan|[+-]?inf)',text,re.I)
                    dump(RUN/f'gpu{gpu}_{dataset}.json',result)
                    finished.append(result)
                    if result['status']!='passed':raise RuntimeError(f'Worker stopped before target batch: {result["status"]}')
                    print('DONE',gpu,dataset,'peak',max(x['peak_reserved_mib'] for x in result['stages']),flush=True)
                    if queues[gpu]:launch(gpu)
            time.sleep(.5)
    except BaseException as exc:
        failure=repr(exc)
        raise
    finally:
        for child,dataset,start,log in active.values():
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            log.close()
        dump(RUN/'summary.json',{'status':'failed' if failure else 'passed','failure':failure,'results':finished,'formal_training':False,'temporary_checkpoint_dirs_removed':not any(RUN.glob('temporary-checkpoint-*'))})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--gpu',choices=['0','1']);parser.add_argument('--dataset',choices=DATASETS)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run_checks()
    else:
        try:worker(args.gpu,args.dataset)
        except BaseException:traceback.print_exc();raise
