"""CPU correctness checks; no GPU initialization or formal experiments."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stateFormer.flat_transformer import FlatTransGenerator
from stateFormer.flat_train import flat_nll, save_flat_checkpoint, load_flat_checkpoint


class FlatTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = FlatTransGenerator(2, 3, 3, [3, 4, 5], hidden_dim=32,
                                        embedding_dim=8, noise_dim=4,
                                        transformer_layers=1, nhead=4, dropout=0).eval()
        self.seq = torch.tensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                                 [[2, 1, 5], [1, 4, 2], [0, 0, 0]]])
        self.labels = torch.tensor([0, 1])
        self.lengths = torch.tensor([3, 2])
        self.noise = torch.randn(2, 9, 4)

    def test_attribute_shift_and_previous_field_embedding(self):
        inputs = self.model.shift_targets(self.seq)
        self.assertEqual(inputs[0].tolist(), [0, 1, 2, 3, 2, 3, 4, 3, 4])
        seen = {}
        hooks = []
        for field, emb in enumerate(self.model.value_embeddings):
            hooks.append(emb.register_forward_pre_hook(
                lambda module, args, field=field: seen.update({field: args[0].clone()})
            ))
        self.model(self.labels, self.lengths, inputs, self.noise)
        for h in hooks:
            h.remove()
        for field in range(3):
            torch.testing.assert_close(seen[field], inputs[:, field+1::3])

    def test_causal_no_target_or_future_noise_leakage(self):
        inputs = self.model.shift_targets(self.seq)
        original = self.model(self.labels, self.lengths, inputs, self.noise)["logits"]
        changed = self.seq.clone().flatten(1)
        for position in range(4, 9):
            changed[:, position] = changed[:, position] % self.model.x_list[position % 3] + 1
        changed = changed.reshape_as(self.seq)
        noise = self.noise.clone()
        noise[:, 5:] += 20
        altered = self.model(self.labels, self.lengths, self.model.shift_targets(changed), noise)["logits"]
        for position in range(5):
            torch.testing.assert_close(original[position % 3][:, position // 3],
                                       altered[position % 3][:, position // 3], atol=1e-6, rtol=1e-6)

    def test_prefix_prediction_matches_teacher_forcing(self):
        inputs = self.model.shift_targets(self.seq)
        with torch.no_grad():
            full = self.model(self.labels, self.lengths, inputs, self.noise)["logits"]
            for position in range(6):
                next_logits = self.model.next_logits(self.labels, self.lengths, inputs[:, :position+1], self.noise[:, :position+1])
                torch.testing.assert_close(next_logits, full[position % 3][:, position // 3], atol=1e-5, rtol=1e-5)

    def test_masked_weighted_loss_and_noise_gradient(self):
        self.model.train()
        outputs = self.model(self.labels, self.lengths, self.model.shift_targets(self.seq), self.noise)
        weights = torch.tensor([0.7, 1.3])
        loss = flat_nll(outputs, self.seq, self.lengths, weights)
        manual = sum(F.cross_entropy(outputs["logits"][f][b, :n], self.seq[b, :n, f], reduction="sum") * weights[b]
                     for f in range(3) for b, n in enumerate([3, 2])) / 3
        torch.testing.assert_close(loss, manual)
        padded = self.seq.clone()
        padded[1, 2] = torch.tensor([3, 4, 5])
        torch.testing.assert_close(loss, flat_nll(outputs, padded, self.lengths, weights))
        loss.backward()
        self.assertGreater(float(self.model.noise_projection.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_fixed_noise_sampling_valid_ids_and_prefix(self):
        observed = []
        original = self.model._hidden
        def checked(labels, lengths, inputs, noise):
            torch.testing.assert_close(noise, self.noise[:, :inputs.shape[1]])
            observed.append(inputs.shape[1])
            return original(labels, lengths, inputs, noise)
        self.model._hidden = checked
        sample = self.model.sample(2, self.labels, self.lengths, noise=self.noise)
        self.assertEqual(observed, list(range(1, 10)))
        for field, bins in enumerate(self.model.x_list):
            self.assertTrue(((sample[:, :, field] >= 1) & (sample[:, :, field] <= bins)).all())
        prefix = sample[:, :1]
        continuation = self.model.sample(2, self.labels, self.lengths, x=prefix, noise=self.noise)
        torch.testing.assert_close(continuation[:, :1], prefix)

    def test_checkpoint_identity_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pth"
            metadata = {"fields": ["a", "b", "c"], "bins_sha256": "test"}
            save_flat_checkpoint(path, self.model, metadata)
            loaded, actual = load_flat_checkpoint(path, "cpu", metadata)
            loaded.eval()
            inputs = self.model.shift_targets(self.seq)
            for a, b in zip(self.model(self.labels, self.lengths, inputs, self.noise)["logits"],
                            loaded(self.labels, self.lengths, inputs, self.noise)["logits"]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                load_flat_checkpoint(path, "cpu", {"fields": ["wrong"]})
            torch.save(self.model.state_dict(), path)
            with self.assertRaises(ValueError):
                load_flat_checkpoint(path, "cpu")

    def test_complete_chimerge_train_select_generate_entry(self):
        from flat_driver import run_flat
        with tempfile.TemporaryDirectory(prefix=".flat-entry-check-", dir=ROOT) as folder:
            root = Path(folder)
            config = json.loads((ROOT / "config.json").read_text())
            config["model_type"] = FlatTransGenerator.model_type
            config["binning"] = {"method": "chimerge"}
            config["flat_model"] = dict(hidden_dim=32, embedding_dim=8, noise_dim=4,
                                         transformer_layers=1, nhead=4, dropout=0)
            config["model_paras"].update(epoch=1, checkpoint=1, batch_size=2, max_seq_len=3, expand_times=0)
            config["path"]["dataset"] = "tiny"
            for key in ("json", "bins", "model", "result"):
                config["path"][key + "_folder"] = str((root / key).relative_to(ROOT))
            inputs = root / "json/tiny"
            inputs.mkdir(parents=True)
            original = ROOT / "log/hybrid16_gpu_smoke_20261120_v1/input/iscx"
            for source in sorted(original.glob("*.json"))[:2]:
                flows = copy.deepcopy(json.loads(source.read_text())[:2])
                for i, flow in enumerate(flows):
                    flow["series"] = flow["series"][:3 if i == 0 else 1]
                (inputs / source.name).write_text(json.dumps(flows))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = run_flat(config, device="cpu")
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(list(result["generated_counts"].values()), [2, 2])
            for path in (root / "result/tiny").glob("*.json"):
                flows = json.loads(path.read_text())
                self.assertEqual([len(f["series"]) for f in flows], [3, 1])
            with self.assertRaises(FileExistsError):
                run_flat(config, device="cpu")
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
