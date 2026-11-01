"""Standalone checkpoint evaluation on bpRNA or RNAStrAlign/ArchiveII/PK cohorts."""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import hashlib
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bgmfold.evaluation.decoding import (
    decode_curve,
    has_pseudoknot,
    load_record,
    metrics,
    summarize,
    unordered_pairs,
)
from bgmfold.io import bundle_reports, write_csv, write_json
from bgmfold.threshold_grid import default_threshold_candidates

ROOT = Path(__file__).resolve().parents[2]


def dataset_jobs(benchmark, data_root):
    d = Path(data_root)
    if benchmark == "bprna":
        return [
            ("bprna", "val", d / "bprna-spot0/validation.parquet"),
            ("bprna", "test", d / "bprna-spot0/test.parquet"),
        ]
    return [
        ("rnastralign", "val", d / "rnastralign.600/validation.parquet"),
        ("rnastralign", "test", d / "rnastralign.600/test.parquet"),
        ("archiveii", "test", d / "archiveii.600/test.parquet"),
    ]


def resolve_task(model, state, task="auto"):
    model = copy.deepcopy(model)
    has_p = any(k.startswith("pairedness_head.") for k in state)
    has_t = any(k.startswith("pair_type_head.") for k in state)
    inferred = {
        (False, False): "contact",
        (True, False): "pairedness",
        (False, True): "pairtype",
        (True, True): "multitask",
    }[has_p, has_t]
    if task != "auto" and task != inferred:
        raise ValueError(f"Requested {task}, checkpoint heads indicate {inferred}")
    tasks = model["multitask"]
    tasks["enabled"] = has_p or has_t
    tasks["pairedness"]["enabled"] = has_p
    tasks["pair_type"]["enabled"] = has_t
    return model, inferred


def infer(
    checkpoint,
    model_config,
    cache,
    jobs,
    batch_size=1,
    amp=True,
    device="cuda:0",
    task="auto",
):
    import torch

    from bgmfold.data.datasets import RNADataset, collate_fn
    from bgmfold.models.discriminative_contact_map_model import (
        build_discriminative_contact_map_model,
    )
    from bgmfold.utils import _checkpoint_model_state

    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is required for GPU checkpoint inference")
    torch.manual_seed(2528)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    state, _ = _checkpoint_model_state(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model_cfg = yaml.safe_load(Path(model_config).read_text())
    model_cfg, task = resolve_task(model_cfg, state, task)
    path = Path(model_cfg["rna_encoder_path"])
    model_cfg["rna_encoder_path"] = str(path if path.is_absolute() else ROOT / path)
    model_cfg["rna_encoder_initialize_from_config_only"] = True
    model = build_discriminative_contact_map_model(model_cfg)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval().requires_grad_(False)
    model.rnafm.set_runtime_freeze(True)
    audit = dict(
        checkpoint=str(Path(checkpoint).resolve()),
        task=task,
        strict_load=True,
        device=device,
        batch_size=batch_size,
        amp=amp,
        torch_version=torch.__version__,
        model=model_cfg,
        datasets=[],
    )
    for dataset, split, path in jobs:
        raw = RNADataset([str(path)], max_length=None)
        if raw.df.id.astype(str).duplicated().any():
            raise ValueError(f"Duplicate IDs: {dataset}/{split}")
        if not (
            raw.df.sequence.str.len() == raw.df.secondary_structure.str.len()
        ).all():
            raise ValueError(f"Sequence/GT length mismatch: {dataset}/{split}")
        loader = torch.utils.data.DataLoader(
            raw,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        folder = cache / dataset / split
        folder.mkdir(parents=True, exist_ok=True)
        for step, batch in enumerate(loader):
            with (
                torch.inference_mode(),
                torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=amp and device.startswith("cuda"),
                ),
            ):
                logits, mask = model(batch["sequences"])
            for index, sequence in enumerate(batch["sequences"]):
                n = len(sequence)
                if int(mask[index].sum()) != n:
                    raise ValueError(f"Encoder mask mismatch: {batch['ids'][index]}")
                matrix = logits[index, :n, :n].float().cpu().numpy()
                if not np.isfinite(matrix).all() or not np.array_equal(
                    matrix, matrix.T
                ):
                    raise ValueError("Expected finite symmetric model logits")
                i, j = np.triu_indices(n, 1)
                identity = str(batch["ids"][index])
                file = folder / (hashlib.sha256(identity.encode()).hexdigest() + ".npz")
                temporary = file.with_suffix(".partial")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        id=np.array(identity),
                        sequence=np.array(sequence),
                        split=np.array(split),
                        dataset=np.array(dataset),
                        contact_logits_upper=matrix[i, j],
                        gt_cells=np.argwhere(
                            batch["contact_map"][index, :n, :n].numpy() > 0.5
                        ).astype(np.int32),
                    )
                temporary.replace(file)
            if step % 100 == 0:
                logging.info(
                    "%s/%s %d/%d RNAs",
                    dataset,
                    split,
                    min((step + 1) * batch_size, len(raw)),
                    len(raw),
                )
        audit["datasets"].append(
            dict(dataset=dataset, split=split, n=len(raw), path=str(path))
        )
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return audit


def scan_record(job):
    path, grid = job
    r = load_record(path)
    counts, _ = decode_curve(r, grid)
    return dict(
        path=str(path),
        id=r["id"],
        length=len(r["sequence"]),
        gt_pairs=len(unordered_pairs(r["gt_cells"])),
        has_gt_pseudoknot=has_pseudoknot(r["gt_cells"]),
        counts=counts,
    )


def analyze(cache, report, jobs, benchmark, workers=8):
    grid = default_threshold_candidates()
    scanned = {}
    summaries = []
    mapper = map
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        mapper = pool.map
    try:
        for dataset, split, path in jobs:
            paths = sorted((cache / dataset / split).glob("*.npz"))
            expected = len(pd.read_parquet(path, columns=["id"]))
            if len(paths) != expected:
                raise ValueError(
                    f"Incomplete cache {dataset}/{split}: {len(paths)}/{expected}"
                )
            scanned[dataset, split] = list(
                mapper(scan_record, ((p, grid) for p in paths))
            )
    finally:
        if pool:
            pool.shutdown()
    for protocol in (
        ("full", "prifold_lt490") if benchmark == "rnastralign" else ("full",)
    ):
        groups = {
            k: [r for r in rs if protocol == "full" or r["length"] < 490]
            for k, rs in scanned.items()
        }
        val = groups[("bprna" if benchmark == "bprna" else "rnastralign", "val")]
        if not val:
            raise ValueError("The validation length group is empty")
        val_f1 = (
            metrics(np.stack([r["counts"] for r in val]).reshape(-1, 3))[:, 2]
            .reshape(len(val), len(grid))
            .mean(axis=0)
        )
        selected = max(range(len(grid)), key=lambda i: (val_f1[i], grid[i]))
        threshold = grid[selected]
        for (dataset, split), records in groups.items():
            folder = report / protocol / dataset / split
            folder.mkdir(parents=True, exist_ok=True)
            rows = []
            curve = []
            full_counts = []
            for i, t in enumerate(grid):
                c = np.asarray(
                    [r["counts"][i] for r in records], dtype=np.int64
                ).reshape(-1, 3)
                m = metrics(c) if len(c) else np.empty((0, 3))
                curve.append(
                    dict(
                        threshold=t,
                        n=len(records),
                        precision=float(m[:, 0].mean()) if len(m) else None,
                        recall=float(m[:, 1].mean()) if len(m) else None,
                        f1=float(m[:, 2].mean()) if len(m) else None,
                        f1_empty_one=float(
                            np.where(c.sum(axis=1) == 0, 1.0, m[:, 2]).mean()
                        )
                        if len(m)
                        else None,
                    )
                )
            with gzip.open(
                folder / "pair_evidence.jsonl.gz", "wt", encoding="utf-8"
            ) as evidence:
                for r in records:
                    tp, fp, fn = map(int, r["counts"][selected])
                    p, rec, f = metrics(r["counts"][selected : selected + 1])[0]
                    rows.append(
                        dict(
                            id=r["id"],
                            length=r["length"],
                            threshold=threshold,
                            gt_pairs=r["gt_pairs"],
                            predicted_pairs=(tp + fp) // 2,
                            tp=tp,
                            fp=fp,
                            fn=fn,
                            precision=float(p),
                            recall=float(rec),
                            f1=float(f),
                            f1_empty_one=1.0 if tp + fp + fn == 0 else float(f),
                            has_gt_pseudoknot=r["has_gt_pseudoknot"],
                        )
                    )
                    full_counts.append(
                        dict(
                            id=r["id"], length=r["length"], counts=r["counts"].tolist()
                        )
                    )
                    raw = load_record(r["path"])
                    _, pairs = decode_curve(raw, [threshold], return_pairs=True)
                    evidence.write(
                        json.dumps(
                            dict(
                                id=r["id"],
                                sequence=raw["sequence"],
                                threshold=threshold,
                                gt_cells=raw["gt_cells"].tolist(),
                                gt_pairs=unordered_pairs(raw["gt_cells"]).tolist(),
                                predicted_pairs=pairs,
                                pair_index_base=0,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            fields = [
                "id",
                "length",
                "threshold",
                "gt_pairs",
                "predicted_pairs",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "f1_empty_one",
                "has_gt_pseudoknot",
            ]
            write_csv(folder / "threshold_scan.csv", curve)
            write_csv(folder / "per_rna.csv", rows, fields)
            write_json(
                folder / "per_rna_threshold_counts.json",
                dict(
                    thresholds=grid,
                    count_columns=["tp", "fp", "fn"],
                    count_unit="matrix_cell",
                    records=full_counts,
                ),
            )
            result = dict(
                dataset=dataset,
                split=split,
                length_protocol=protocol,
                cohort="all",
                threshold=threshold,
                **summarize(rows),
            )
            summaries.append(result)
            write_json(folder / "summary.json", result)
            if benchmark == "rnastralign" and split == "test":
                pk = [r for r in rows if r["has_gt_pseudoknot"]]
                pk_records = [r for r in records if r["has_gt_pseudoknot"]]
                write_csv(folder / "pseudoknot_per_rna.csv", pk, fields)
                pk_curve = []
                for i, t in enumerate(grid):
                    c = np.asarray(
                        [r["counts"][i] for r in pk_records], dtype=np.int64
                    ).reshape(-1, 3)
                    m = metrics(c) if len(c) else np.empty((0, 3))
                    pk_curve.append(
                        dict(
                            threshold=t,
                            n=len(pk),
                            f1=float(m[:, 2].mean()) if len(m) else None,
                            f1_empty_one=float(
                                np.where(c.sum(axis=1) == 0, 1.0, m[:, 2]).mean()
                            )
                            if len(m)
                            else None,
                        )
                    )
                write_csv(folder / "pseudoknot_threshold_scan.csv", pk_curve)
                summaries.append(
                    dict(
                        dataset=dataset,
                        split=split,
                        length_protocol=protocol,
                        cohort="gt_pseudoknot_RNAs_complete_structure",
                        threshold=threshold,
                        **summarize(pk),
                    )
                )
            logging.info(
                "%s %s/%s threshold=%.3f macro-F1=%s",
                protocol,
                dataset,
                split,
                threshold,
                result["per_rna_f1"],
            )
    write_csv(report / "all_results.csv", summaries)
    write_json(
        report / "summary.json",
        dict(
            results=summaries,
            threshold_selection="Val macro F1; higher threshold on exact tie; Test scans diagnostic only",
            scoring="Every true-length L x L cell, including diagonal GT, no padding; strict positions, no shift tolerance",
            decoding="Symmetric stable raw-logit greedy, no self-pairs, all other distances allowed, no auxiliary gating",
            pseudoknots="GT crossing i<k<j<l defines cohort; score complete RNA structure at the same Val-selected threshold, not crossing-edge-only F1",
            f1_empty_one="Only GT and prediction both empty score one; GT empty with nonempty prediction still scores zero",
        ),
    )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("bprna", "rnastralign"), required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Path or name=path; repeat to evaluate several checkpoints sequentially on one GPU",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "contact", "pairedness", "pairtype", "multitask"),
        default="auto",
    )
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs/model_gbrna.yaml"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable BF16 autocast; model parameters already stay FP32",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Rescore existing local caches without importing torch",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error("Batch size and workers must be positive")
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.analyze_only:
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output / "logs/evaluation.log"),
        ],
    )
    jobs = dataset_jobs(args.benchmark, args.data_root)
    manifest = dict(benchmark=args.benchmark, status="running", models={})
    labels = set()
    code = 0
    for i, item in enumerate(args.checkpoint):
        label, path = (
            item.split("=", 1)
            if "=" in item and not Path(item).is_file()
            else (("model" if len(args.checkpoint) == 1 else f"model{i + 1}"), item)
        )
        if Path(label).name != label or label in ("", ".", "..") or label in labels:
            parser.error("Use unique checkpoint labels without path separators")
        labels.add(label)
        cache = args.output / "cache" / label
        report = args.output / "reports" / label
        try:
            if not args.analyze_only:
                audit = infer(
                    Path(path),
                    args.model_config,
                    cache,
                    jobs,
                    args.batch_size,
                    not args.fp32,
                    args.device,
                    args.task,
                )
                write_json(report / "inference.json", audit)
                config_dir = args.output / "configs"
                config_dir.mkdir(exist_ok=True)
                (config_dir / f"{label}.yaml").write_text(
                    yaml.safe_dump(
                        dict(
                            model=audit["model"],
                            benchmark=args.benchmark,
                            device=args.device,
                            batch_size=args.batch_size,
                            amp=not args.fp32,
                            task=audit["task"],
                        ),
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
            analyze(cache, report, jobs, args.benchmark, args.workers)
            manifest["models"][label] = dict(status="completed", checkpoint=path)
        except Exception as exc:
            code = 1
            manifest["models"][label] = dict(
                status="failed", checkpoint=path, error=str(exc)
            )
            logging.exception("Checkpoint %s failed", label)
    manifest.update(
        status="completed" if code == 0 else "partial_or_failed", exit_code=code
    )
    write_json(args.output / "reports/manifest.json", manifest)
    path = bundle_reports(args.output)
    logging.info("Finished: %s; report=%s", manifest["status"], path)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
