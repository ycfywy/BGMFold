"""One pretrained encoder, multiple datasets, unchanged partner/stem metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from bgmfold.attention_structure_metrics import (
    DEFAULT_STEM_MIN_SEPARATION,
    layer_metrics,
    prepare_structure,
)
from bgmfold.data.dotbracket import BRACKET_PAIRS, dotbracket_to_contact_map

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("partner_rank", "stem_recovery_1")
BOOTSTRAP_METRICS = set(METRICS)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_encoder(kind, source, config_path, device, precision):
    """Read a local model DIRECTORY shard by shard; require full encoder coverage."""
    import torch

    from bgmfold.gbrna_weight_directory import directory_plan
    from bgmfold.vendor.rnabert.configuration_rnabert import RNABertConfig
    from bgmfold.vendor.rnabert.modeling_rnabert import RNABertModel
    from bgmfold.vendor.rnabert.tokenization_rnabert import RNABertTokenizer

    plan = directory_plan(source, config_path)
    print(
        f"[{kind}] config={plan['config']}; config_fallback={plan['config_fallback']}",
        flush=True,
    )
    config = RNABertConfig.from_pretrained(
        str(Path(plan["config"]).parent), local_files_only=True
    )
    model = RNABertModel(config, add_pooling_layer=False)
    expected = set(model.state_dict())
    loaded, all_keys, skipped, weight_files = set(), set(), [], []
    for filename in plan["files"]:
        path = Path(filename)
        print(f"[{kind}] loading shard {path.name}", flush=True)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            shard = load_file(str(path), device="cpu")
        else:
            shard = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict):
            raise ValueError(f"Expected flat tensor dictionary: {path}")
        selected = {}
        for key, value in shard.items():
            if key in all_keys:
                raise ValueError(f"Duplicate tensor across shards: {key}")
            all_keys.add(key)
            if plan["weight_map"] is not None and plan["weight_map"].get(key) != str(
                path.relative_to(Path(plan["source"]))
            ):
                raise ValueError(f"Tensor disagrees with shard index: {key}")
            if not torch.is_tensor(value):
                raise ValueError(f"Non-tensor weight entry: {key}")
            encoder_key = key.removeprefix("bert.")
            if encoder_key in expected:
                if encoder_key in loaded:
                    raise ValueError(
                        f"Duplicate encoder tensor after prefix removal: {key}"
                    )
                selected[encoder_key] = value
                loaded.add(encoder_key)
            elif key.startswith("cls."):
                skipped.append(
                    key
                )  # MLM head is explicitly outside these attention diagnostics.
            else:
                raise ValueError(f"Unrecognized model tensor: {key}")
        # Each call is partial only because the file is sharded. Shapes are checked
        # here; full missing/unexpected coverage is enforced across ALL shards below.
        result = model.load_state_dict(selected, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected encoder tensors: {result.unexpected_keys}")
        weight_files.append(
            dict(name=path.name, size=path.stat().st_size, sha256=digest_file(path))
        )
        del selected, shard
        gc.collect()
    if loaded != expected:
        raise ValueError(f"Incomplete encoder: missing {sorted(expected - loaded)}")
    if plan["weight_map"] is not None and all_keys != set(plan["weight_map"]):
        raise ValueError("Actual shard tensor keys do not match weight index")
    audit = dict(
        kind=kind,
        source=plan["source"],
        config=plan["config"],
        config_fallback=plan["config_fallback"],
        config_sha256=digest_file(plan["config"]),
        index=plan["index"],
        index_sha256=digest_file(plan["index"]) if plan["index"] else None,
        strict_encoder_load=True,
        encoder_tensors=len(loaded),
        skipped_mlm_tensors=skipped,
        weight_files=weight_files,
    )
    audit.update(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_attention_heads,
    )
    model.requires_grad_(False)
    model.to(dtype=torch.bfloat16 if precision == "bfloat16" else torch.float32)
    model.to(device).eval()
    tokenizer = RNABertTokenizer(
        str(ROOT / "bgmfold/vendor/rnabert/vocab.txt"), version="v2"
    )
    return model, tokenizer, audit


def load_rnafm_encoder(source, device, precision):
    """Load one local MultiMolecule RNA-FM encoder, without a task head."""
    return load_multimolecule_encoder(source, device, precision, "rnafm")


def load_rinalmo_encoder(source, device, precision):
    """Load one local MultiMolecule RiNALMo encoder, without a task head."""
    return load_multimolecule_encoder(source, device, precision, "rinalmo")


def load_multimolecule_encoder(source, device, precision, family):
    """HF-style RNA encoders share loading/reporting but keep their own config."""
    import multimolecule  # noqa: F401 -- registers RNA-FM with AutoModel
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    source = Path(source).expanduser().resolve()
    config = AutoConfig.from_pretrained(str(source), local_files_only=True)
    label = {"rnafm": "RNA-FM", "rinalmo": "RiNALMo"}[family]
    if config.model_type != family:
        raise ValueError(
            f"Expected {label} pretrained directory, got {config.model_type}: {source}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        str(source),
        config=config,
        local_files_only=True,
        attn_implementation="eager",
        add_pooling_layer=False,
        output_loading_info=True,
    )
    # An original pretraining checkpoint can contain a discarded MLM head, but
    # every encoder tensor must load; never analyze randomly initialized layers.
    if any(
        loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"{label} encoder did not load completely: {loading}")
    model.requires_grad_(False)
    model.to(dtype=torch.bfloat16 if precision == "bfloat16" else torch.float32)
    model.to(device).eval()
    # Transformers versions return list or set values here. Store stable JSON
    # lists so a successful load cannot fail during report writing/resuming.
    loading_info = {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in loading.items()
    }
    audit = dict(
        kind=family,
        source=str(source),
        config=str(source / "config.json"),
        strict_encoder_load=True,
        loading_info=loading_info,
        add_pooling_layer=False,
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_attention_heads,
        attention_implementation="eager",
        tokenizer_input="unspaced_RNA",
        parameter_dtype=precision,
    )
    return model, tokenizer, audit


def encoder_forward_inputs(model, sequences):
    """GB.RNA uses spaced tokens and a vendor-only layer-selection argument."""
    if model.config.model_type in ("rnafm", "rinalmo"):
        return list(sequences), {}
    return [" ".join(sequence) for sequence in sequences], dict(
        attention_layer_indices=tuple(range(model.config.num_hidden_layers))
    )


def infer_scores(model, tokenizer, sequences, device, precision):
    """Padded batch forward; return nucleotide-only arrays, one per RNA."""
    import torch

    inputs, model_kwargs = encoder_forward_inputs(model, sequences)
    encoded = tokenizer(inputs, return_tensors="pt", padding=True)
    lengths = [len(sequence) for sequence in sequences]
    if encoded["attention_mask"].sum(dim=1).tolist() != [n + 2 for n in lengths]:
        raise ValueError("Tokenizer must preserve exactly one token per nucleotide")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and precision == "bfloat16",
        ),
    ):
        output = model(
            **encoded, output_attentions=True, return_dict=True, **model_kwargs
        )
        if (
            output.attentions is None
            or len(output.attentions) != model.config.num_hidden_layers
        ):
            raise RuntimeError("All layers must return attention maps")
        # Accumulate head means in float32 without a full float32 heads tensor.
        # Transfer all layers together, rather than synchronizing once per layer.
        means = torch.stack(
            [
                attention.mean(dim=1, dtype=torch.float32)
                for attention in output.attentions
            ]
        )
        raw = means.cpu().numpy()
    return [raw[:, i, 1 : n + 1, 1 : n + 1].copy() for i, n in enumerate(lengths)]


def infer_with_oom_retry(model, tokenizer, batch, device, precision, batch_cap):
    """Yield successful subbatches; retain a smaller cap after CUDA OOM."""
    import torch

    if len(batch) > batch_cap[0]:
        cut = batch_cap[0]
        yield from infer_with_oom_retry(
            model, tokenizer, batch[:cut], device, precision, batch_cap
        )
        yield from infer_with_oom_retry(
            model, tokenizer, batch[cut:], device, precision, batch_cap
        )
        return
    failed = False
    try:
        arrays = infer_scores(
            model, tokenizer, [item[2] for item in batch], device, precision
        )
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise
        failed = True
    if failed:
        # Outside except: release the traceback and failed forward's tensors first.
        gc.collect()
        torch.cuda.empty_cache()
        batch_cap[0] = max(1, len(batch) // 2)
        print(f"CUDA OOM: reducing max batch size to {batch_cap[0]}", flush=True)
        yield from infer_with_oom_retry(
            model, tokenizer, batch, device, precision, batch_cap
        )
    else:
        if len(arrays) != len(batch):
            raise RuntimeError("Batch output count mismatch")
        yield batch, arrays


def validate_structure(structure):
    # The shared historical parser silently ignores unmatched brackets; reject them here.
    opens = {a: b for a, b in BRACKET_PAIRS}
    closes = {b: a for a, b in BRACKET_PAIRS}
    stacks = {a: 0 for a in opens}
    for c in structure:
        if c in opens:
            stacks[c] += 1
        elif c in closes:
            a = closes[c]
            stacks[a] -= 1
            if stacks[a] < 0:
                raise ValueError("Unmatched closing GT bracket")
        elif c not in ".-_:,~":
            raise ValueError(f"Unknown GT symbol: {c}")
    if any(stacks.values()):
        raise ValueError("Unmatched opening GT bracket")


def summarize(rows, bootstrap, seed):
    frame = pd.DataFrame(rows)
    summaries = []
    for (version, split, layer), group in frame.groupby(
        ["version", "split", "layer"], sort=False
    ):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            lower, upper = None, None
            if len(values) and bootstrap and metric in BOOTSTRAP_METRICS:
                rng = np.random.default_rng(seed)
                means = []
                for start in range(0, bootstrap, 32):
                    idx = rng.integers(
                        len(values), size=(min(32, bootstrap - start), len(values))
                    )
                    means.extend(values[idx].mean(axis=1).tolist())
                lower, upper = map(float, np.quantile(means, [0.025, 0.975]))
            summaries.append(
                dict(
                    version=version,
                    split=split,
                    layer=int(layer),
                    metric=metric,
                    mean=float(values.mean()) if len(values) else None,
                    std=float(values.std(ddof=1)) if len(values) > 1 else None,
                    n_valid=len(values),
                    n_total=len(group),
                    ci95_low=lower,
                    ci95_high=upper,
                )
            )
    return summaries


def analyze_raw_attention(
    raw,
    sequence,
    structure,
    seed,
    min_stem_length=3,
    stem_min_separation=DEFAULT_STEM_MIN_SEPARATION,
):
    """CPU replay with independent partner/stem candidate definitions."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (len(sequence), len(sequence)):
        raise ValueError("Raw layer attention shape does not match RNA length")
    if len(structure) != len(sequence) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Invalid structure length or attention values")
    validate_structure(structure)
    st = prepare_structure(
        dotbracket_to_contact_map(structure), min_stem_length, stem_min_separation
    )
    tie_order = np.random.default_rng(seed).permutation(len(st["i"]))
    return [
        layer_metrics((0.5 * (matrix + matrix.T))[st["i"], st["j"]], st, tie_order)
        for matrix in raw
    ]


def prepare_dataset(args, identity, version):
    """Prepare data/protocol before loading an encoder or allocating GPU memory."""
    frames, data_audit = {}, {}
    for split, filename in [
        ("train", "train.parquet"),
        ("val", "validation.parquet"),
        ("test", "test.parquet"),
    ]:
        path = args.data_dir / filename
        frame = pd.read_parquet(path)[["id", "sequence", "secondary_structure"]]
        if frame.id.astype(str).duplicated().any():
            raise ValueError(f"Duplicate RNA IDs: {split}")
        data_audit[split] = dict(
            path=str(path.resolve()), sha256=digest_file(path), n_full=len(frame)
        )
        frames[split] = frame.iloc[: args.limit] if args.limit else frame
        data_audit[split]["n_requested"] = len(frames[split])
    protocol = dict(
        schema=6,
        data=data_audit,
        min_separation=1,
        partner_min_separation=1,
        stem_min_separation=args.stem_min_separation,
        min_stem_length=args.min_stem_length,
        candidate_pairs=f"partner: all j!=i; stem: i<j and j-i>={args.stem_min_separation}",
        stem_eligibility="maximal GT stems; exclude whole stem if any edge fails stem distance",
        weight_identity=identity,
        inference_versions=[version],
        encoder_family=args.encoder_family,
        batching=dict(
            max_batch_size=args.max_batch_size,
            max_pair_tokens=args.max_pair_tokens,
            sort="length_ascending",
            cuda_oom="halve_and_retain_smaller_batch_cap",
        ),
        aggregation="head mean then symmetrization; equal RNA macro mean",
        seed=args.seed,
        precision=args.precision,
        layer_index_base=0,
        topk="distance-eligible GT unordered edge count; rank only distance-eligible candidates; seeded tie permutation fixed across layers",
        no_runtime_length_filter=True,
        model_format=f"local_hf_{args.encoder_family}_directories",
        raw_cache=not args.no_raw_cache,
        raw_cache_dtype="float32",
        raw_cache_definition="mean of all heads; full directional nucleotide LxL before symmetrization; excludes special tokens and padding",
        metrics=list(METRICS),
        bootstrap_metrics=sorted(BOOTSTRAP_METRICS),
        metric_code_hashes={
            name: digest_file(ROOT / name)
            for name in (
                "bgmfold/attention_structure_metrics.py",
                "bgmfold/attention_batching.py",
                "bgmfold/gbrna_weight_directory.py",
                "bgmfold/attention/analyze.py",
            )
        },
    )
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(
                "Existing output: use --resume with identical data and protocol, or a new --out"
            )
    atomic_json(protocol_path, protocol)
    data_folder = args.out / "data"
    data_folder.mkdir(exist_ok=True)
    for split, frame in frames.items():
        frame.to_parquet(data_folder / f"{split}.parquet", index=False)
    return frames, protocol


def analyze_dataset(args, version, versions, model, tokenizer, audit, frames, protocol):
    """Analyze one dataset with an already loaded encoder; keep split caches isolated."""
    try:
        atomic_json(
            args.out / "status.json", dict(status="running", versions=[version])
        )
        folder = args.out / version
        audit_path = folder / "encoder_audit.json"
        if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
            raise ValueError(f"Encoder identity changed while resuming: {version}")
        atomic_json(audit_path, audit)
        atomic_json(
            folder / "encoder_config.json",
            json.loads(Path(audit["config"]).read_text()),
        )
        all_rows = []
        from bgmfold.attention_batching import attention_batches

        batch_cap = [args.max_batch_size]
        for split, frame in frames.items():
            pending = []
            completed = 0
            for index, row in enumerate(frame.itertuples(index=False)):
                sequence = str(row.sequence).upper().replace("T", "U")
                structure = str(row.secondary_structure)
                if not sequence or len(sequence) != len(structure):
                    raise ValueError(f"GT/sequence mismatch: {split}/{row.id}")
                validate_structure(structure)
                sample_hash = hashlib.sha256(
                    (str(row.id) + "\0" + sequence + "\0" + structure).encode()
                ).hexdigest()
                record_path = folder / "per_rna" / split / f"{index:06d}.json"
                cache = folder / "raw_attention_cache" / split / f"{index:06d}.npz"
                if args.resume and record_path.exists():
                    record = json.loads(record_path.read_text())
                    if (
                        record["sample_sha256"] != sample_hash
                        or len(record["layers"]) != audit["n_layers"]
                    ):
                        raise ValueError(f"Resume record mismatch: {record_path}")
                    if args.no_raw_cache or cache.is_file():
                        all_rows.extend(record["layers"])
                        completed += 1
                        continue
                pending.append((index, row, sequence, structure, sample_hash))
            for batch in attention_batches(
                pending,
                lambda item: len(item[2]),
                args.max_batch_size,
                args.max_pair_tokens,
            ):
                for successful, arrays in infer_with_oom_retry(
                    model, tokenizer, batch, args.device, args.precision, batch_cap
                ):
                    print(
                        f"[{version}] {split} batch={len(successful)} max_length={max(len(item[2]) for item in successful)}",
                        flush=True,
                    )
                    for (index, row, sequence, structure, sample_hash), raw in zip(
                        successful, arrays
                    ):
                        record_path = folder / "per_rna" / split / f"{index:06d}.json"
                        cache = (
                            folder / "raw_attention_cache" / split / f"{index:06d}.npz"
                        )
                        rng_seed = int(
                            hashlib.sha256(
                                f"{args.seed}:{split}:{sample_hash}".encode()
                            ).hexdigest()[:16],
                            16,
                        )
                        values = analyze_raw_attention(
                            raw,
                            sequence,
                            structure,
                            rng_seed,
                            args.min_stem_length,
                            args.stem_min_separation,
                        )
                        records = []
                        for layer, result in enumerate(values):
                            result = {
                                k: None
                                if isinstance(v, float) and not np.isfinite(v)
                                else v
                                for k, v in result.items()
                            }
                            records.append(
                                dict(
                                    version=version,
                                    split=split,
                                    row=index,
                                    id=str(row.id),
                                    length=len(sequence),
                                    layer=layer,
                                    **result,
                                )
                            )
                        if not args.no_raw_cache:
                            cache.parent.mkdir(parents=True, exist_ok=True)
                            temporary = cache.with_suffix(".partial")
                            with temporary.open("wb") as stream:
                                metadata = dict(
                                    version=version,
                                    split=split,
                                    row=index,
                                    id=str(row.id),
                                    sequence=sequence,
                                    secondary_structure=structure,
                                    min_separation=1,
                                    partner_min_separation=1,
                                    stem_min_separation=args.stem_min_separation,
                                    min_stem_length=args.min_stem_length,
                                    tie_seed=rng_seed,
                                    sample_sha256=sample_hash,
                                    definition=protocol["raw_cache_definition"],
                                )
                                np.savez_compressed(
                                    stream,
                                    attention=raw.astype(np.float32, copy=False),
                                    metadata=json.dumps(metadata, ensure_ascii=False),
                                )
                            temporary.replace(cache)
                        atomic_json(
                            record_path, dict(sample_sha256=sample_hash, layers=records)
                        )
                        all_rows.extend(records)
                        completed += 1
            print(f"[{version}] {split} completed {completed}/{len(frame)}", flush=True)
            all_rows.sort(key=lambda r: (r["split"], r["row"], r["layer"]))
            # Write complete split tables promptly; preserve them if a later stage fails.
            write_csv(
                folder / f"{split}_per_rna_layers.csv",
                [r for r in all_rows if r["split"] == split],
            )
        summary = summarize(all_rows, args.bootstrap, args.seed)
        write_csv(folder / "layer_summary.csv", summary)
        atomic_json(
            folder / "status.json",
            dict(
                status="completed",
                n_records=len(all_rows),
                full_dataset=args.limit is None,
                bootstrap=args.bootstrap,
            ),
        )
        atomic_json(
            args.out / "status.json",
            dict(
                status="completed",
                versions=[version],
                audited_versions=list(versions),
                full_dataset=args.limit is None,
            ),
        )
    except Exception:
        atomic_json(
            args.out / "status.json",
            dict(status="failed", traceback=traceback.format_exc()),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder-family", choices=("gbrna", "rnafm", "rinalmo"), default="gbrna"
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        required=True,
        help="One complete local pretrained encoder directory, not a task best.pt",
    )
    parser.add_argument(
        "--encoder-config", type=Path, default=ROOT / "configs/encoder_gbrna"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        default=[ROOT / "data/bprna-spot0", ROOT / "data/rnastralign.600"],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-pair-tokens", type=int, default=1_000_000)
    parser.add_argument("--min-stem-length", type=int, default=3)
    parser.add_argument("--stem-min-separation", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2528)
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save directional full L x L, head-averaged attention for every layer; large output",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Explicit per-split smoke-test cap; recorded as partial coverage",
    )
    args = parser.parse_args()
    args.no_raw_cache = not args.save_attention
    if (
        min(
            args.max_batch_size,
            args.max_pair_tokens,
            args.min_stem_length,
            args.stem_min_separation,
        )
        < 1
        or args.bootstrap < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Invalid batch or metric settings")
    if len({p.name for p in args.data_dir}) != len(args.data_dir):
        parser.error("Dataset directories need unique names")
    args.out = args.out.expanduser().resolve()
    args.encoder_path = args.encoder_path.expanduser().resolve()
    if args.out.exists() and not args.resume:
        raise FileExistsError("Use a fresh output directory or --resume")
    args.out.mkdir(parents=True, exist_ok=True)
    version = args.encoder_family
    identity = dict(
        inference_model=version,
        source=str(args.encoder_path),
        comparison_method="single_pretrained_encoder",
    )
    plans = []
    index = dict(encoder_family=version, encoder_loads=1, datasets=[])
    for path in args.data_dir:
        current = argparse.Namespace(**vars(args))
        current.data_dir = path
        current.out = args.out / "datasets" / path.name
        frames, protocol = prepare_dataset(current, identity, version)
        record = dict(
            name=path.name,
            status="pending",
            output=str(current.out.relative_to(args.out)),
        )
        index["datasets"].append(record)
        plans.append((current, frames, protocol, record))
    atomic_json(args.out / "dataset_runs.json", index)
    import torch

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        if version == "gbrna":
            model, tokenizer, audit = load_encoder(
                version,
                args.encoder_path,
                args.encoder_config,
                args.device,
                args.precision,
            )
        else:
            model, tokenizer, audit = load_multimolecule_encoder(
                args.encoder_path, args.device, args.precision, version
            )
        for current, frames, protocol, record in plans:
            record["status"] = "running"
            atomic_json(args.out / "dataset_runs.json", index)
            analyze_dataset(
                current, version, [version], model, tokenizer, audit, frames, protocol
            )
            record["status"] = "completed"
            atomic_json(args.out / "dataset_runs.json", index)
        atomic_json(
            args.out / "status.json",
            dict(
                status="completed",
                datasets=index["datasets"],
                full_dataset=args.limit is None,
            ),
        )
    except Exception:
        atomic_json(
            args.out / "status.json",
            dict(status="failed", traceback=traceback.format_exc()),
        )
        raise
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
