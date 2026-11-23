"""Equal-width and quantile baselines sharing the model's bins JSON format."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from pre_process.ChiMerge import construct_data_dic

DEFAULT_BIN_COUNTS = {
    "time": 100, "pkt_len": 500, "flags": 10, "ttl": 60,
    "src_port": 250, "dst_port": 250, "src_ip": 100, "dst_ip": 100,
}


def value_counts(data_dic, min_value, max_value):
    counts = Counter()
    for data_list in data_dic.values():
        for values in data_list:
            for value, count in values.items():
                if not math.isfinite(value) or count < 0:
                    raise ValueError("Binning requires finite values and nonnegative counts")
                if count:
                    counts[min(max(value, min_value), max_value)] += count
    if not counts:
        raise ValueError("Cannot fit bins to an empty attribute")
    return counts


def _validate(min_unit, min_value, max_value, num_bins):
    if not all(math.isfinite(x) for x in (min_unit, min_value, max_value)):
        raise ValueError("Binning parameters must be finite")
    if min_unit <= 0 or min_value > max_value:
        raise ValueError("Invalid binning range or min_unit")
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")


def _intervals(edges, min_unit):
    # Match the existing dataset's two-decimal lookup and first-match boundaries.
    precision = min(2, max(0, -int(math.floor(math.log10(min_unit)))))
    edges = sorted(set(round(float(x), precision) for x in edges))
    if len(edges) == 1:
        return [[edges[0], edges[0]]]
    return [[left, right] for left, right in zip(edges, edges[1:])]


def cal_bins_equal_width(data_dic, min_unit, min_value, max_value, num_bins=20):
    """Split the configured range into at most num_bins rounded intervals."""
    _validate(min_unit, min_value, max_value, num_bins)
    return _intervals(np.linspace(min_value, max_value, num_bins + 1), min_unit)


def cal_bins_equal_freq(data_dic, min_unit, min_value, max_value, num_bins=20):
    """Linear quantiles equivalent to np.quantile on counts-expanded values."""
    _validate(min_unit, min_value, max_value, num_bins)
    counts = value_counts(data_dic, min_value, max_value)
    values = np.array(sorted(counts), dtype=float)
    cumulative = np.cumsum([counts[v] for v in values], dtype=np.int64)
    ranks = np.linspace(0, 1, num_bins + 1) * (int(cumulative[-1]) - 1)
    low = values[np.searchsorted(cumulative, np.floor(ranks).astype(np.int64), side="right")]
    high = values[np.searchsorted(cumulative, np.ceil(ranks).astype(np.int64), side="right")]
    edges = low + (high - low) * (ranks - np.floor(ranks))
    return _intervals(edges, min_unit)


def basemerge(dataset, json_folder, bins_folder, ip_attrs, port_attrs, sery_attrs,
              max_seq_len, params_dic, method="equal_width", bin_counts=None):
    methods = {"equal_width": cal_bins_equal_width, "equal_freq": cal_bins_equal_freq}
    if method not in methods:
        raise ValueError(f"Unsupported baseline method: {method}")
    counts = dict(DEFAULT_BIN_COUNTS)
    counts.update(bin_counts or {})
    attrs = sery_attrs + port_attrs + ip_attrs
    if list(params_dic) != attrs:
        raise ValueError("Attribute and bin parameter orders must match")
    for attr in attrs:
        _validate(**params_dic[attr], num_bins=counts[attr])
    json_data = {}
    for path in sorted((Path(json_folder) / dataset).glob("*.json")):
        json_data[path.name.split(".")[0]] = json.loads(path.read_text())
    data = construct_data_dic(json_data, attrs, port_attrs, ip_attrs, sery_attrs, max_seq_len)
    result = {}
    for attr in attrs:
        # Validate even equal-width inputs; an empty attribute is an input error.
        value_counts(data[attr], params_dic[attr]["min_value"], params_dic[attr]["max_value"])
        intervals = methods[method](data[attr], **params_dic[attr], num_bins=counts[attr])
        print(f"{method} {attr}: requested={counts[attr]}, actual={len(intervals)}")
        result[attr] = {"intervals": intervals}
    output = Path(bins_folder) / f"bins_{dataset}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False))
    return result
