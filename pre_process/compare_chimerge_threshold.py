"""CPU-only binning comparison; imports no model/training/GPU modules."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep the historical ablation reproducible without changing the main ChiMerge.
ABLATION_SOURCE = ROOT / 'log/chimerge_threshold_ablation_20261118_v1/ChiMerge.py'
_ablation_spec = importlib.util.spec_from_file_location('threshold_ablation_chimerge', ABLATION_SOURCE)
ChiMerge = importlib.util.module_from_spec(_ablation_spec)
_ablation_spec.loader.exec_module(ChiMerge)

DATASETS = ('iscx', 'ssrc', 'mirage', 'utmn', 'cesnet3')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def summarize(intervals, counts, min_unit):
    values = np.array(sorted(counts), dtype=float)
    weights = np.array([counts[v] for v in values], dtype=np.int64)
    hits = np.zeros(len(values), dtype=np.int64)
    assigned = np.full(len(values), -1, dtype=np.int64)
    occupancy = np.zeros(len(intervals), dtype=np.int64)
    for i, (start, end) in enumerate(intervals):
        found = (values >= start) & (values <= end)
        first = found & (assigned < 0)
        assigned[first] = i
        hits += found
    if intervals:
        # Dataset uses edge-bin fallback only outside first/last endpoints.
        assigned[(assigned < 0) & (values < intervals[0][0])] = 0
        assigned[(assigned < 0) & (values > intervals[-1][1])] = len(intervals) - 1
    for i, count in zip(assigned, weights):
        if i >= 0:
            occupancy[i] += count
    widths = np.array([end-start for start,end in intervals], dtype=float)
    result = dict(bin_count=len(intervals), invalid_intervals=sum(a>b for a,b in intervals),
        singleton_bins=sum(a==b for a,b in intervals), empty_bins=int((occupancy==0).sum()),
        observed_unique=len(values), observations=int(weights.sum()),
        outside_bins_observations=int(weights[hits==0].sum()),
        unencodable_gap_observations=int(weights[assigned<0].sum()),
        overlapping_observations=int(weights[hits>1].sum()),
        max_bin_share=float(occupancy.max()/weights.sum()) if len(occupancy) else None,
        width_median=float(np.median(widths)) if len(widths) else None,
        width_p90=float(np.quantile(widths,.9)) if len(widths) else None,
        width_max=float(widths.max()) if len(widths) else None)
    return result


def check_switch(legacy_path):
    spec = importlib.util.spec_from_file_location('legacy_chimerge', legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    fixtures = [([{0:5, .01:1, .03:1}, {0:4,.01:1,.03:1}], .01,0,1),
                ([{0:20,1:1,3:1,5:20},{0:10,1:1,3:1,5:15}],1,0,10)]
    with np.errstate(invalid='ignore', divide='ignore'):
        for data, unit, low, high in fixtures:
            a, ad = legacy.cal_bins_one_category(data,unit,low,high)
            b, bd = ChiMerge.cal_bins_one_category(data,unit,low,high)
            assert a == b
            np.testing.assert_array_equal(ad,bd)
            merged = {'a':data,'b':data}
            assert legacy.cal_bins(merged,unit,low,high) == ChiMerge.cal_bins(merged,unit,low,high)
        original = ChiMerge.threshold_merge
        def forbidden(*args, **kwargs):
            raise AssertionError('threshold_merge invoked in disabled branch')
        ChiMerge.threshold_merge = forbidden
        try:
            for data,unit,low,high in fixtures:
                ChiMerge.cal_bins_one_category(data,unit,low,high,use_threshold_merge=False)
                ChiMerge.cal_bins({'a':data},unit,low,high,use_threshold_merge=False)
        finally:
            ChiMerge.threshold_merge = original
    return 'passed: default matches original, disabled path never calls threshold_merge'


def run(run_id):
    os.chdir(ROOT)
    logdir = ROOT / 'log' / run_id
    binsroot = ROOT / 'bins' / run_id
    assert not logdir.exists() and not binsroot.exists(), 'Use a new run ID'
    logdir.mkdir(parents=True)
    binsroot.mkdir(parents=True)
    # The pre-existing frozen copy is used only to verify default compatibility.
    legacy = ROOT / 'log/binning_formal_20261117_v1/source/pre_process/ChiMerge.py'
    check = check_switch(legacy)
    (logdir/'ChiMerge.py').write_bytes(Path(ChiMerge.__file__).read_bytes())
    (logdir/'compare_chimerge_threshold.py').write_bytes(Path(__file__).read_bytes())
    base = json.loads((ROOT/'config.json').read_text())
    manifest = dict(run_id=run_id, started=datetime.now().astimezone().isoformat(),
                    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                    training=False, switch_check=check, datasets={})
    manifest['source_sha256']={name: hashlib.sha256((logdir/name).read_bytes()).hexdigest()
                               for name in ('ChiMerge.py','compare_chimerge_threshold.py')}
    allrows=[]
    write_json(logdir/'status.json',dict(status='running',stage='preparing',pid=os.getpid()))
    for dataset in DATASETS:
        print('START',dataset,flush=True)
        tick=time.perf_counter()
        attrs=base['attributes']
        sery=list(attrs['sery_attribute']); ports=list(attrs['port_attribute']); ips=list(attrs['ip_attribute'])
        params={**attrs['sery_attribute'],**attrs['port_attribute'],**attrs['ip_attribute']}
        counts={attr:Counter() for attr in params}; hashes={}; flows_per_class={}
        for path in sorted((ROOT/'data-json'/dataset).glob('*.json')):
            raw=path.read_bytes(); hashes[path.name]=hashlib.sha256(raw).hexdigest()
            flows=json.loads(raw); flows_per_class[path.stem]=len(flows)
            for flow in flows:
                for attr in ports+ips:
                    counts[attr][round(flow[attr],2)] += 1
                for packet in flow['series'][:16]:
                    for attr in sery:
                        counts[attr][round(packet[attr],2)] += 1
            del flows,raw
        assert flows_per_class
        details={}; bins={}
        cfg=dict(dataset=dataset,json_folder='data-json',max_seq_len=16,attributes=attrs)
        write_json(logdir/f'{dataset}.config.json',cfg)
        for variant,enabled in (('original',True),('no_threshold',False)):
            write_json(logdir/'status.json',dict(status='running',dataset=dataset,variant=variant,pid=os.getpid()))
            folder=binsroot/variant; folder.mkdir(exist_ok=True)
            start=time.perf_counter()
            with (logdir/f'{dataset}.{variant}.log').open('x') as out, contextlib.redirect_stdout(out), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('default')
                ChiMerge.chimerge(dataset,'data-json',str(folder.relative_to(ROOT)),ips,ports,sery,16,params,
                                  use_threshold_merge=enabled)
            bins[variant]=json.loads((folder/f'bins_{dataset}.json').read_text())
            details[variant]=dict(seconds=time.perf_counter()-start,
                warnings=[dict(message=str(w.message),line=w.lineno) for w in caught],fields={})
            for attr in params:
                details[variant]['fields'][attr]=summarize(bins[variant][attr]['intervals'],counts[attr],params[attr]['min_unit'])
            print(dataset,variant,'seconds=',round(details[variant]['seconds'],2),flush=True)
        rows=[]
        for attr in params:
            a=details['original']['fields'][attr]; b=details['no_threshold']['fields'][attr]
            row=dict(dataset=dataset,field=attr,original_bins=a['bin_count'],no_threshold_bins=b['bin_count'],
                     delta=b['bin_count']-a['bin_count'],
                     ratio=b['bin_count']/a['bin_count'] if a['bin_count'] else None,
                     identical_intervals=bins['original'][attr]['intervals']==bins['no_threshold'][attr]['intervals'])
            for variant,detail in details.items():
                for key in ('singleton_bins','empty_bins','width_median','width_p90','width_max',
                            'outside_bins_observations','unencodable_gap_observations','overlapping_observations','invalid_intervals','max_bin_share'):
                    row[f'{variant}_{key}']=detail['fields'][attr][key]
            rows.append(row)
        allrows+=rows
        manifest['datasets'][dataset]=dict(input_sha256=hashes,flow_counts=flows_per_class,
            seconds=time.perf_counter()-tick,variants=details)
        write_json(logdir/f'{dataset}.comparison.json',rows)
        write_json(logdir/'manifest.json',manifest)
        print('DONE',dataset,[(r['field'],r['original_bins'],r['no_threshold_bins']) for r in rows],flush=True)
    with (logdir/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    write_json(logdir/'comparison.json',allrows)
    report=['# ChiMerge threshold_merge CPU ablation','',check,'',
            'Same eight fields, first 16 packets, configured ranges; only threshold_merge is disabled.',
            'Counts use observed values and the existing first-match / outside-edge fallback semantics.',
            'IP/port occupancy counts each flow once; sequence fields count each retained packet.',
            'No training or GPU modules imported. This compares discretization, not model quality.','',
            '| Dataset | Field | Original bins | No threshold | Delta | Ratio | Identical intervals |',
            '|---|---|---:|---:|---:|---:|---|']
    for r in allrows:
        report.append(f"| {r['dataset']} | {r['field']} | {r['original_bins']} | {r['no_threshold_bins']} | {r['delta']:+} | {r['ratio']:.3f} | {r['identical_intervals']} |")
    (logdir/'REPORT.md').write_text('\n'.join(report)+'\n')
    write_json(logdir/'status.json',dict(status='completed',datasets=list(DATASETS),comparisons=len(allrows),training=False,cuda_visible_devices='-1'))
    print('COMPLETED',logdir/'REPORT.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='chimerge_threshold_ablation_20261118_v1')
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():
        parser.error('run-id must contain only letters, digits and underscores')
    try:
        run(args.run_id)
    except BaseException as exc:
        status=ROOT/'log'/args.run_id/'status.json'
        if status.exists():
            write_json(status,dict(status='failed',error=repr(exc),training=False,cuda_visible_devices='-1'))
        raise
