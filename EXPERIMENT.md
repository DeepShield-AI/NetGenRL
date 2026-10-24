# Baseline Transformer Experiment

## Workflow

Run the pipeline from the repository root with the `exp` Conda environment. The driver discovers labels from JSON filenames in the selected dataset directory, then:

1. Encodes flows using `FlowByteDataset` and trains the byte-level causal Transformer.
2. Saves one checkpoint every 50 completed epochs under `model/baseline-transformer/<dataset>/epoch_<epoch>.pth`; early stopping is disabled by default so all requested epochs are reached. It also saves final and best-validation checkpoints in that dataset directory.
3. Uses the checkpoint with the lowest validation loss saved by `train.py` (`traffic_transformer_best.pth`). No OT evaluation or bins file is required.
4. Generates one output per input flow using that checkpoint and writes `<label>.json` under `result/baseline-transformer/<dataset>/`. The JSON field structure follows `post_process/data_save.save_data`: flow-level metadata and a `series` list containing packet attributes.

The Transformer uses raw byte-level traffic fields and does not read the bins. Generated records retain the `data_save` field layout, which omits protocol and other source fields not represented by that serializer.

## Run

From the repository root, train for 1000 epochs on each dataset:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n exp python baseline/transformer/driver.py --data data-json/mirage --epochs 1000
CUDA_VISIBLE_DEVICES=1 conda run -n exp python baseline/transformer/driver.py --data data-json/mirage-train --epochs 1000
CUDA_VISIBLE_DEVICES=1 conda run -n exp python baseline/transformer/driver.py --data data-json/utmn --epochs 1000
CUDA_VISIBLE_DEVICES=1 conda run -n exp python baseline/transformer/driver.py --data data-json/utmn-train --epochs 1000
```

The process and result paths are selected from the input directory name. For example, `mirage-train` writes under `model/baseline-transformer/mirage-train/` and `result/baseline-transformer/mirage-train/`. The four commands expose only physical GPU 1.

With `CUDA_VISIBLE_DEVICES=1`, PyTorch sees the selected physical GPU as `cuda:0`. If CUDA is unavailable, the existing model config selects CPU.

Useful options:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n exp python baseline/transformer/driver.py --data data-json/utmn --epochs 1000 --eval-samples 20
```

`--eval-samples` and `--bins` remain accepted for command-line compatibility but are unused. `--max-packets` caps generated packet counts and defaults to 16, matching the repository configuration. `--patience N` enables early stopping after N non-improving validation epochs; generation always uses the lowest-validation-loss checkpoint saved before stopping. Checkpoint spacing remains 50 epochs for archival checkpoints.

## Validation Notes

The `exp` environment has PyTorch 2.7.1 and NumPy. The baseline driver uses CPU when CUDA is unavailable.
