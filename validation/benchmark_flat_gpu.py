"""Bounded GPU1 benchmark of flat token model; never launch formal experiments."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'log/flat_transformer_smoke_20261120_v1'
PYTHON='/data/xiaozj/anaconda3/envs/exp/bin/python'
DATASETS=('iscx','ssrc','mirage','utmn','cesnet3')


def dump(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)


def prepare():
    RUN.mkdir(parents=True,exist_ok=False)
    manifest=dict(gpu=1,max_packets=16,max_tokens=128,hidden_dim=512,layers=4,
        batches=[1,2,4,8,16,32,64],warmup_steps=1,measured_steps=3,
        allocator_fraction=.5,used_limit_mib=6144,free_min_mib=4096,temperature_limit=85,timeout_per_worker_seconds=240,
        formal_training=False,source_sha256={})
    files=['stateFormer/flat_transformer.py','stateFormer/flat_train.py','stateFormer/dataset.py',
           'stateFormer/transformer.py','post_process/flat_pipeline.py','post_process/check_bestmodel.py',
           'post_process/data_save.py','pre_process/ChiMerge.py','flat_driver.py']
    for name in files:
        source=ROOT/name;target=RUN/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes());manifest['source_sha256'][name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (RUN/'original_transformer.py').write_bytes((ROOT/'log/binning_formal_20261117_v1/source/stateFormer/transformer.py').read_bytes())
    for dataset in DATASETS:
        src=ROOT/'log/hybrid16_gpu_smoke_20261120_v1/input'/dataset
        shutil.copytree(src,RUN/'input'/dataset)
        bins=ROOT/'bins/chimerge_threshold_ablation_20261118_v1/original'/f'bins_{dataset}.json'
        target=RUN/'bins'/bins.name;target.parent.mkdir(exist_ok=True)
        target.write_bytes(bins.read_bytes())
    dump(RUN/'manifest.json',manifest)


def worker(dataset,kind):
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    sys.path.insert(0,str(RUN/'source'))
    import torch
    from torch.nn import functional as F
    from stateFormer.flat_transformer import FlatTransGenerator
    from stateFormer.flat_train import flat_nll
    from stateFormer.dataset import CustomDataset
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(.5,0)
    torch.manual_seed(20261120)
    labels={p.stem:i for i,p in enumerate(sorted((RUN/'input'/dataset).glob('*.json')))}
    bins=RUN/'bins'/f'bins_{dataset}.json'
    counts=[len(v['intervals']) for v in json.loads(bins.read_text()).values()]
    data=CustomDataset(str(RUN/'input'/dataset)+'/',str(bins),labels,16,['src_port','dst_port','src_ip','dst_ip'],['time','pkt_len','flags','ttl'],preencode=True)
    seqs,label_tensor,lengths,weights=data._encoded
    longest=torch.where(lengths==16)[0];assert len(longest)>0
    if kind=='flat':
        model=FlatTransGenerator(len(labels),8,16,counts).cuda()
    else:
        spec=importlib.util.spec_from_file_location('original_packet_model',RUN/'original_transformer.py')
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        model=original.TransGenerator(len(labels),8,16,counts,torch.device('cuda')).cuda()
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001,betas=(.5,.999))
    report=dict(dataset=dataset,kind=kind,gpu=1,samples=len(data),classes=len(labels),total_bins=sum(counts),
                max_packets=16,max_tokens=128 if kind=='flat' else 16,parameters=parameters,stages=[],status='running')
    path=RUN/f'{kind}_{dataset}.json';dump(path,report)
    previous_peak=None
    for batch in ([1,2,4,8,16,32,64] if kind=='flat' else [8,64]):
        if previous_peak and previous_peak*2>4.5*2**30:
            report.update(status='budget_limited',next_batch=batch);dump(path,report);return
        indices=longest[torch.arange(batch)%len(longest)]
        x,y,l,w=[v[indices].cuda() for v in (seqs,label_tensor,lengths,weights)]
        def step():
            model.train();optimizer.zero_grad()
            if kind=='flat':
                loss=flat_nll(model(y,l,model.shift_targets(x)),x,l,w)
            else:
                seed=torch.cat([x.new_zeros((batch,1,8)),x[:,:-1]],dim=1)
                out=model(y,l,seed,x);mask=torch.arange(16,device='cuda')[None]<l[:,None]
                nll=x.new_zeros((),dtype=torch.float32);offset=0
                for field,size in enumerate(model.output_sizes):
                    target=x[:,:,field].masked_fill(~mask,1)
                    value=F.nll_loss(out['preds'][:,:,offset:offset+size].reshape(-1,size),target.reshape(-1),reduction='none').reshape(batch,16)
                    nll=nll+(value*mask*w[:,None]).sum();offset+=size
                kl=(-.5*(1+out['logvar']-out['mu'].pow(2)-out['logvar'].exp()).sum(-1)*mask*w[:,None]).sum()
                loss=nll/8+.01*kl
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            loss.backward();optimizer.step()
            return loss
        torch.cuda.reset_peak_memory_stats()
        step();torch.cuda.synchronize() # one warmup, also allocate Adam state
        timings=[]
        for _ in range(3):
            start=time.perf_counter();loss=step();torch.cuda.synchronize();timings.append(time.perf_counter()-start)
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        stage=dict(batch=batch,seconds_per_step=timings,median_step_seconds=statistics.median(timings),loss=float(loss.detach()),
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        previous_peak=torch.cuda.max_memory_reserved();report['stages'].append(stage);dump(path,report);print('STAGE',stage,flush=True)
    # Measure autoregressive generation separately on one flow. Reuse fixed
    # positional noise inside each flat sample; generation output is not saved.
    model.eval();torch.cuda.reset_peak_memory_stats();generation=[]
    with torch.no_grad():
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            fake=model.sample(1,label_tensor[:1].cuda(),torch.tensor([16.],device='cuda'))
            torch.cuda.synchronize();generation.append(time.perf_counter()-start)
            assert fake.shape==(1,16,8)
            for field,count in enumerate(counts):assert ((fake[:,:,field]>=1)&(fake[:,:,field]<=count)).all().item()
    report.update(status='passed',verified_batch=report['stages'][-1]['batch'],generation_seconds=generation,
                  generation_median_seconds=statistics.median(generation),
                  generation_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  generation_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    dump(path,report);print('PASSED',kind,dataset,flush=True)


def gpu_state():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
    return {line.split(',')[0].strip():list(map(int,line.split(',')[1:])) for line in raw.splitlines()}


def run():
    manifest=json.loads((RUN/'manifest.json').read_text())
    for relative,digest in manifest['source_sha256'].items():assert hashlib.sha256((RUN/'source'/relative).read_bytes()).hexdigest()==digest
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=5)
    uuid=next(row.split(',')[1].strip() for row in rows.splitlines() if row.split(',')[0].strip()=='1')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=5)
    assert uuid not in apps,'GPU1 is busy; do not disturb other tasks'
    results=[];failure=None
    try:
        for kind,dataset in [('flat',d) for d in DATASETS]+[('packet','mirage')]:
            with (RUN/f'{kind}_{dataset}.log').open('x') as log:
                child=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'worker','--dataset',dataset,'--kind',kind],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),stdout=log,stderr=subprocess.STDOUT)
                print('START',kind,dataset,'physical GPU1 PID',child.pid,flush=True)
                start=time.monotonic();monitor=dict(peak_gpu1_used_mib=0,minimum_gpu1_free_mib=10240,peak_gpu1_temperature_c=0)
                try:
                    while True:
                        code=child.poll()
                        if code is not None:break
                        state=gpu_state();used,free,temp=state['1']
                        monitor['peak_gpu1_used_mib']=max(monitor['peak_gpu1_used_mib'],used)
                        monitor['minimum_gpu1_free_mib']=min(monitor['minimum_gpu1_free_mib'],free)
                        monitor['peak_gpu1_temperature_c']=max(monitor['peak_gpu1_temperature_c'],temp)
                        if used>6144 or free<4096 or temp>=85 or time.monotonic()-start>240:
                            raise RuntimeError(f'Bounded verification safety stop: used={used}, free={free}, temperature={temp}')
                        # Avoid adding heat if the neighboring ongoing GPU0 job
                        # is already at the configured shutdown threshold.
                        if state['0'][2]>=85:raise RuntimeError('GPU0 temperature reached 85C; stopping only GPU1 verification')
                        time.sleep(.5)
                    if code!=0:raise RuntimeError(f'Worker exit {code}: {kind}/{dataset}')
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
            result=json.loads((RUN/f'{kind}_{dataset}.json').read_text());result['monitor']=monitor
            dump(RUN/f'{kind}_{dataset}.json',result);results.append(result)
            if result['status']!='passed':raise RuntimeError('Stopped at memory budget before target batch')
            print('DONE',kind,dataset,'batch',result['verified_batch'],flush=True)
    except BaseException as exc:
        failure=repr(exc);raise
    finally:
        dump(RUN/'summary.json',dict(status='failed' if failure else 'passed',failure=failure,results=results,formal_training=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--dataset',choices=DATASETS);parser.add_argument('--kind',choices=['flat','packet'],default='flat');args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run()
    else:
        try:worker(args.dataset,args.kind)
        except BaseException:traceback.print_exc();raise
