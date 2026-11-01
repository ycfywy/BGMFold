"""CPU regression checks; never import/download torch or load model weights."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml
from bgmfold.attention.analyze import analyze_raw_attention
from bgmfold.attention_structure_metrics import prepare_structure
from bgmfold.evaluation.decoding import (
    decode_curve,
    dense_counts,
    has_pseudoknot,
    metrics,
)
from bgmfold.evaluation.run import analyze, resolve_task
from bgmfold.training.run import completed_role, configure

ROOT = Path(__file__).resolve().parents[1]


def core_signature(node):
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        # Empty type_params was added in Python 3.12; ignore that version-only
        # field so the source-equivalence check also works on Python 3.10/3.11.
        if getattr(item, "type_params", None) == []:
            item._fields = tuple(f for f in item._fields if f != "type_params")
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def record(n=8, gt=()):
    return dict(
        sequence="A" * n,
        contact_logits_upper=np.full(n * (n - 1) // 2, -20.0),
        gt_cells=np.asarray(gt, dtype=int).reshape(-1, 2),
    )


def set_score(r, i, j, value):
    n = len(r["sequence"])
    r["contact_logits_upper"][i * (2 * n - i - 1) // 2 + j - i - 1] = value


class CPUChecks(unittest.TestCase):
    def test_full_map_counts_include_asymmetric_and_diagonal_gt(self):
        r = record(gt=[(0, 1), (1, 0), (2, 2), (3, 5)])
        set_score(r, 0, 1, 3)
        set_score(r, 3, 5, 2)
        counts, pairs = decode_curve(r, [0.5], True)
        dense = np.zeros((8, 8), bool)
        gt = np.zeros((8, 8), bool)
        for i, j in pairs:
            dense[i, j] = dense[j, i] = True
        for i, j in r["gt_cells"]:
            gt[i, j] = True
        np.testing.assert_array_equal(counts[0], dense_counts(dense, gt))
        np.testing.assert_array_equal(counts[0], [3, 1, 1])

    def test_adjacent_pairs_and_stable_raw_logit_ties(self):
        r = record(gt=[(0, 1), (1, 0)])
        set_score(r, 0, 1, 0)
        c, p = decode_curve(r, [0.5], True)
        self.assertEqual(p, [[0, 1]])
        np.testing.assert_array_equal(c[0], [2, 0, 0])
        set_score(r, 0, 2, 0)
        self.assertEqual(decode_curve(r, [0.5], True)[1], [[0, 1]])
        set_score(r, 0, 1, 20)
        set_score(r, 0, 2, 21)
        self.assertEqual(decode_curve(r, [0.999], True)[1], [[0, 2]])

    def test_empty_case_and_pseudoknot_definition(self):
        self.assertFalse(has_pseudoknot([(0, 6), (1, 5)]))
        self.assertTrue(has_pseudoknot([(0, 4), (2, 6)]))
        r = record()
        np.testing.assert_array_equal(decode_curve(r, [0.5])[0], [[0, 0, 0]])
        self.assertEqual(metrics(np.array([[0, 0, 0]]))[0, 2], 0)

    def test_scan_matches_independent_dense_greedy(self):
        rng = np.random.default_rng(2528)
        grid = [0.9, 0.2, 0.5, 0.99]
        for n in (1, 2, 8, 16):
            for _ in range(8):
                r = record(n)
                r["contact_logits_upper"] = rng.integers(
                    -3, 4, len(r["contact_logits_upper"])
                ).astype(float)
                g = rng.random((n, n)) < 0.15
                r["gt_cells"] = np.argwhere(g)
                counts, _ = decode_curve(r, grid)
                ii, jj = np.triu_indices(n, 1)
                for k, t in enumerate(grid):
                    pred = np.zeros((n, n), bool)
                    used = set()
                    for index in np.argsort(-r["contact_logits_upper"], kind="stable"):
                        a, b = int(ii[index]), int(jj[index])
                        score = r["contact_logits_upper"][index]
                        if score < np.log(t / (1 - t)):
                            break
                        if a not in used and b not in used:
                            used.update((a, b))
                            pred[a, b] = pred[b, a] = True
                    np.testing.assert_array_equal(counts[k], dense_counts(pred, g))

    def test_completed_or_early_stopped_role_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "checkpoints").mkdir()
            job = dict(
                directory=folder,
                settings={
                    "train": {
                        "num_epochs": 400,
                        "early_stopping": {"enabled": True, "patience": 4},
                    }
                },
            )
            for epoch, bad, expected in [
                (399, 0, "max_epochs"),
                (99, 4, "validation_patience"),
                (99, 3, None),
            ]:
                (folder / "checkpoints/training_state.json").write_text(
                    json.dumps({"checkpoint_epoch": epoch, "bad_eval_count": bad})
                )
                self.assertEqual(completed_role(job), expected)

    def test_loss_implementations_are_source_equivalent(self):
        expected = json.loads((ROOT / "docs/core_equivalence.json").read_text())[
            "loss_function_ast"
        ]
        module = ast.parse((ROOT / "bgmfold/losses.py").read_text())
        for n in module.body:
            if isinstance(n, ast.FunctionDef) and n.name in expected:
                self.assertEqual(
                    core_signature(n),
                    expected.pop(n.name),
                )
        self.assertFalse(expected)

    def test_model_definitions_preserve_checkpoint_architecture(self):
        expected = json.loads((ROOT / "docs/core_equivalence.json").read_text())[
            "model_definitions"
        ]
        for filename, definitions in expected.items():
            text = (
                (ROOT / "bgmfold/models" / filename)
                .read_text()
                .replace("bgmfold", "symfold")
                .replace("BGMFold", "SymFold")
            )
            tree = ast.parse(text)
            actual = {
                n.name: core_signature(n)
                for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            }
            self.assertEqual(actual, definitions)

    def test_cptm_roles_and_training_recipe(self):
        cfg = yaml.safe_load((ROOT / "configs/train_bprna_cptm.yaml").read_text())
        for role, p, t in [
            ("contact", False, False),
            ("pairedness", True, False),
            ("pairtype", False, True),
            ("multitask", True, True),
        ]:
            c = configure(
                cfg, role, Path("/encoder"), Path("/out") / role, ROOT / "data"
            )
            self.assertEqual(c["model"]["multitask"]["pairedness"]["enabled"], p)
            self.assertEqual(c["model"]["multitask"]["pair_type"]["enabled"], t)
            self.assertEqual(
                c["loss"]["multitask"]["pairedness_weight"], 0.1 if p else 0
            )
            self.assertEqual(
                c["loss"]["multitask"]["pair_type_weight"], 0.1 if t else 0
            )
            self.assertEqual(c["loss"]["pos_weight"], 300)
            self.assertEqual(c["loss"]["normalization"], "per_sequence")
            self.assertEqual(c["train"]["num_epochs"], 400)
            self.assertEqual(c["train"]["early_stopping"]["patience"], 4)
            self.assertNotIn("max_length", c["data"])
        r = yaml.safe_load(
            (ROOT / "configs/train_rnastralign_multitask.yaml").read_text()
        )
        self.assertEqual(r["data"]["batch_size_by_bucket"][6], 12)
        self.assertEqual(r["data"]["batch_size_by_bucket"][18], 2)
        self.assertEqual(r["train"]["num_epochs"], 400)
        for state, task in [
            ({}, "contact"),
            ({"pairedness_head.weight": 0}, "pairedness"),
            ({"pair_type_head.0.weight": 0}, "pairtype"),
            ({"pairedness_head.weight": 0, "pair_type_head.0.weight": 0}, "multitask"),
        ]:
            self.assertEqual(resolve_task(cfg["model"], state)[1], task)

    def test_cache_analysis_full_scan_and_strict_length_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            jobs = []
            for dataset, split in [
                ("rnastralign", "val"),
                ("rnastralign", "test"),
                ("archiveii", "test"),
            ]:
                rows = []
                for index, n in enumerate((8, 490)):
                    identity = f"{dataset}-{split}-{index}"
                    gt = [(0, 4), (4, 0), (2, 6), (6, 2)] if index == 0 else []
                    r = record(n, gt)
                    if index == 0:
                        set_score(r, 0, 4, 2)
                        set_score(r, 2, 6, 2)
                    folder = cache / dataset / split
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        folder / f"{index}.npz",
                        id=np.array(identity),
                        sequence=np.array(r["sequence"]),
                        dataset=np.array(dataset),
                        split=np.array(split),
                        contact_logits_upper=r["contact_logits_upper"],
                        gt_cells=r["gt_cells"],
                    )
                    rows.append(dict(id=identity))
                path = root / f"{dataset}_{split}.parquet"
                pd.DataFrame(rows).to_parquet(path, index=False)
                jobs.append((dataset, split, path))
            summary = analyze(cache, root / "reports", jobs, "rnastralign", workers=1)
            full = [
                r
                for r in summary
                if r["length_protocol"] == "full"
                and r["dataset"] == "rnastralign"
                and r["split"] == "test"
                and r["cohort"] == "all"
            ][0]
            self.assertEqual(full["n"], 2)
            self.assertEqual(full["per_rna_f1"], 0.5)
            self.assertEqual(full["per_rna_f1_empty_one"], 1)
            pk = [r for r in summary if r["cohort"] != "all"]
            self.assertTrue(all(r["n"] == 1 and r["per_rna_f1"] == 1 for r in pk))
            short = [r for r in summary if r["length_protocol"] == "prifold_lt490"]
            self.assertTrue(all(r["n"] == 1 for r in short))
            table = pd.read_csv(
                root / "reports/full/rnastralign/test/threshold_scan.csv"
            )
            self.assertEqual(len(table), 110)
            data = json.loads(
                (
                    root / "reports/full/rnastralign/test/per_rna_threshold_counts.json"
                ).read_text()
            )
            self.assertEqual(len(data["records"]), 2)
            self.assertEqual(len(data["records"][0]["counts"]), 110)

            # Exercise the public CPU-only CLI, including spawn workers and ZIP.
            from bgmfold.evaluation import run as evaluation

            output = root / "cli"
            shutil.copytree(cache, output / "cache/model")
            checkpoint = root / "epoch=40.pt"
            checkpoint.write_bytes(b"not read during analyze-only")
            argv = [
                "evaluate",
                "--benchmark",
                "rnastralign",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--analyze-only",
                "--workers",
                "2",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(evaluation, "dataset_jobs", return_value=jobs),
                patch.object(
                    evaluation,
                    "infer",
                    side_effect=AssertionError("No GPU inference expected"),
                ),
            ):
                with self.assertRaises(SystemExit) as completed:
                    evaluation.main()
            self.assertEqual(completed.exception.code, 0)
            with ZipFile(output / "analysis_bundle.zip") as archive:
                self.assertIn(
                    "reports/model/full/rnastralign/test/pair_evidence.jsonl.gz",
                    archive.namelist(),
                )
                self.assertFalse(
                    any(x.startswith("cache/") for x in archive.namelist())
                )

            bp_cache = root / "bp_cache"
            bp_jobs = []
            for _, split, path in jobs[:2]:
                folder = bp_cache / "bprna" / split
                folder.mkdir(parents=True)
                for file in (cache / "rnastralign" / split).glob("*.npz"):
                    with np.load(file, allow_pickle=False) as data:
                        values = {k: data[k].copy() for k in data.files}
                    values["dataset"] = np.array("bprna")
                    np.savez_compressed(folder / file.name, **values)
                bp_jobs.append(("bprna", split, path))
            bp_summary = analyze(
                bp_cache, root / "bp_reports", bp_jobs, "bprna", workers=1
            )
            self.assertEqual(len(bp_summary), 2)
            self.assertTrue(all(row["length_protocol"] == "full" for row in bp_summary))

    def test_attention_keeps_distinct_partner_and_stem_rules(self):
        target = np.zeros((8, 8), bool)
        target[0, 1] = target[1, 0] = True
        st = prepare_structure(target, min_stem_length=1, stem_min_separation=4)
        # Short GT is retained for partner ranking, not silently deleted globally.
        self.assertTrue(st is not None)
        result = analyze_raw_attention(
            np.ones((2, 8, 8)) / 8,
            "A" * 8,
            "()......",
            2528,
            min_stem_length=1,
            stem_min_separation=4,
        )
        self.assertEqual(len(result), 2)

    def test_dataset_copies_and_no_parent_imports(self):
        for item in json.loads((ROOT / "data/datasets.json").read_text()):
            path = ROOT / "data" / item["dataset"] / (item["split"] + ".parquet")
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"]
            )
            frame = pd.read_parquet(path)
            self.assertEqual(len(frame), item["n"])
        for path in (ROOT / "bgmfold").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom) and n.module:
                    self.assertFalse(
                        n.module.startswith(("symfold", "scripts")), (path, n.module)
                    )
                    if n.module.startswith("bgmfold"):
                        module = ROOT / Path(*n.module.split("."))
                        self.assertTrue(
                            module.with_suffix(".py").is_file()
                            or (module / "__init__.py").is_file(),
                            (path, n.module),
                        )

    def test_relocated_checkout_cli_and_dry_run_without_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            clone = Path(temporary) / "release"
            shutil.copytree(
                ROOT,
                clone,
                ignore=shutil.ignore_patterns("__pycache__", "outputs", "*.egg-info"),
            )
            env = dict(os.environ, PYTHONPATH=str(clone))
            for module in [
                "bgmfold.training.run",
                "bgmfold.evaluation.run",
                "bgmfold.attention.analyze",
            ]:
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=temporary,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bgmfold.training.run",
                    "--experiment",
                    "bprna",
                    "--encoder-path",
                    "/placeholder/encoder",
                    "--output",
                    str(Path(temporary) / "planned"),
                    "--dry-run",
                ],
                cwd=temporary,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.strip().splitlines()), 4)
            cfg = yaml.safe_load(
                (Path(temporary) / "planned/contact/configs/training.yaml").read_text()
            )
            self.assertTrue(str(clone) in cfg["data"]["train_datasets"][0]["path"])


if __name__ == "__main__":
    unittest.main()
