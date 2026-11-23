"""Bounded baseline checks; model execution requires physical GPU 0."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile

# Set visibility before importing torch or the training entry point.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pre_process import baseline


def check_bins():
    rng = np.random.default_rng(20261117)
    for _ in range(100):
        values = rng.integers(-20, 21, size=50)
        counts = {int(v): int((values == v).sum()) for v in set(values)}
        clipped = np.clip(values, -10, 10)
        expected = np.unique(np.round(np.quantile(clipped, np.linspace(0, 1, 11)), 2))
        actual = baseline.cal_bins_equal_freq({'x': [counts]}, .01, -10, 10, 10)
        np.testing.assert_allclose([actual[0][0]] + [v[1] for v in actual], expected)
    for fn in (baseline.cal_bins_equal_width, baseline.cal_bins_equal_freq):
        assert fn({'x': [{0: 100}]}, 1, 0, 0, 60) == [[0.0, 0.0]]
    assert baseline.cal_bins_equal_width({}, 1, -10, 10, 4) == [[-10, -5], [-5, 0], [0, 5], [5, 10]]
    try:
        baseline.cal_bins_equal_freq({}, 1, 0, 10)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty input was accepted')
    print('PASS: quantile reference, constant fields, width boundaries, empty input')


def check_pipeline():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 0 unavailable; model smoke will not fall back to CPU')
    from driver import driver
    from stateFormer.dataset import CustomDataset
    print('Smoke device:', torch.cuda.get_device_name(0), 'CUDA_VISIBLE_DEVICES=0')
    for dataset in ('iscx', 'ssrc', 'mirage', 'utmn', 'cesnet3'):
        base = json.loads(Path(f'configs/binning_20261117_v1/equal_width_{dataset}.json').read_text())
        # Read only two classes and retain two real flows from each.
        samples = {p.name: json.loads(p.read_text())[:2]
                   for p in sorted((Path('data-json') / dataset).glob('*.json'))[:2]}
        assert len(samples) == 2 and all(samples.values())
        with tempfile.TemporaryDirectory(prefix='.binning-smoke-', dir='.') as folder:
            root = Path(folder)
            inputs = root / 'input' / dataset
            inputs.mkdir(parents=True)
            for name, flows in samples.items():
                (inputs / name).write_text(json.dumps(flows))
            for method in ('equal_width', 'equal_freq'):
                cfg = copy.deepcopy(base)
                cfg['binning']['method'] = method
                cfg['path']['json_folder'] = str(root / 'input')
                for kind in ('bins', 'model', 'result'):
                    cfg['path'][kind + '_folder'] = str(root / method / kind)
                cfg['model_paras'].update(epoch=1, checkpoint=1, batch_size=4, expand_times=0)
                # Only this temporary config runs one bounded smoke epoch.
                driver(cfg)
                result = root / method / 'result' / dataset
                for name, original in samples.items():
                    flows = json.loads((result / name).read_text())
                    assert len(flows) == len(original)
                    for flow, source in zip(flows, original):
                        assert len(flow['series']) == min(16, len(source['series']))
                        assert all(math.isfinite(flow[k]) for k in ('src_ip', 'dst_ip', 'src_port', 'dst_port'))
                        assert all(math.isfinite(v) for p in flow['series'] for v in p.values())
                fields = list(cfg['attributes']['sery_attribute'])
                meta = list(cfg['attributes']['port_attribute']) + list(cfg['attributes']['ip_attribute'])
                mapping = {Path(name).stem: i for i, name in enumerate(sorted(samples))}
                encoded = CustomDataset(str(inputs) + '/', str(root / method / 'bins' / f'bins_{dataset}.json'), mapping, 16, meta, fields)
                for seq, _, length, _ in encoded:
                    assert (seq[:int(length)] >= 1).all() and (seq[int(length):] == 0).all()
                try:
                    driver(cfg)
                except FileExistsError:
                    pass
                else:
                    raise AssertionError('Fresh-start output guard failed')
                print(f'PASS: {dataset}/{method}, training/selection/generation/IDs/fresh-start')
    print('PASS: all ten bounded GPU 0 pipeline cases; temporary outputs removed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit-only', action='store_true')
    args = parser.parse_args()
    check_bins()
    if not args.unit_only:
        check_pipeline()
